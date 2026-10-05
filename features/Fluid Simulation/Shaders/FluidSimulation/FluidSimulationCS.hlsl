// Position-Based Fluids solver (Macklin & Mueller 2013) with Akinci 2013 cohesion.
// Each entry point is one pass; FluidSimulation.cpp dispatches them in order.
// Every pass uses the same register layout so the passes can share one file.

#include "Common/FrameBuffer.hlsli"
#include "Common/SharedData.hlsli"
#include "FluidSimulation/FluidSimulationCommon.hlsli"

StructuredBuffer<Particle> ParticlesIn : register(t0);
StructuredBuffer<float4> PredictedIn : register(t1);
StructuredBuffer<uint> CellCountsIn : register(t2);
StructuredBuffer<uint> CellParticlesIn : register(t3);
StructuredBuffer<float2> LambdasIn : register(t4);  // x: lambda, y: density
StructuredBuffer<float4> VelocitiesIn : register(t5);
Texture2D<float> SceneDepth : register(t6);

RWStructuredBuffer<Particle> Particles : register(u0);
RWStructuredBuffer<float4> PredictedOut : register(u1);
RWStructuredBuffer<uint> CellCounts : register(u2);
RWStructuredBuffer<uint> CellParticles : register(u3);
RWStructuredBuffer<float2> LambdasOut : register(u4);
RWStructuredBuffer<float4> VelocitiesOut : register(u5);

// Camera-relative position of the scene surface at a depth texel
float3 ReconstructSurface(int2 texel, float depth, float2 renderSize)
{
	float2 uv = (float2(texel) + 0.5) / renderSize;
	float2 ndc = uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
	float4 position = mul(FrameBuffer::CameraViewProjInverse, float4(ndc, depth, 1.0));
	return position.xyz / position.w;
}

// Offset that pushes a camera-relative position out of the visible scene surface under it,
// or zero when the position is off-screen, against the sky, or clear of the surface
float3 DepthCollisionOffset(float3 positionCR)
{
	float3 offset = 0.0;

	float4 clip = mul(FrameBuffer::CameraViewProj, float4(positionCR, 1.0));
	float2 ndc = clip.xy / max(clip.w, 1e-6);
	[branch] if (clip.w > 0.0 && all(abs(ndc) < 1.0))
	{
		float2 renderSize = SharedData::BufferDim.xy * FrameBuffer::DynamicResolutionParams1.xy;
		float2 uv = ndc * float2(0.5, -0.5) + 0.5;
		int2 texel = clamp(int2(uv * renderSize), 1, int2(renderSize) - 2);

		float depth = SceneDepth.Load(int3(texel, 0));
		[branch] if (depth < 0.99999)  // skip the sky
		{
			// Surface normal from neighboring texels, taking whichever side is continuous with
			// the center texel so object silhouettes don't produce sideways normals
			float depthLeft = SceneDepth.Load(int3(texel - int2(1, 0), 0));
			float depthRight = SceneDepth.Load(int3(texel + int2(1, 0), 0));
			float depthUp = SceneDepth.Load(int3(texel - int2(0, 1), 0));
			float depthDown = SceneDepth.Load(int3(texel + int2(0, 1), 0));

			float3 center = ReconstructSurface(texel, depth, renderSize);
			float3 tangentX = abs(depthRight - depth) < abs(depth - depthLeft) ?
			                      ReconstructSurface(texel + int2(1, 0), depthRight, renderSize) - center :
			                      center - ReconstructSurface(texel - int2(1, 0), depthLeft, renderSize);
			float3 tangentY = abs(depthDown - depth) < abs(depth - depthUp) ?
			                      ReconstructSurface(texel + int2(0, 1), depthDown, renderSize) - center :
			                      center - ReconstructSurface(texel - int2(0, 1), depthUp, renderSize);

			float3 normal = cross(tangentX, tangentY);
			float normalLength = length(normal);
			if (normalLength > 1e-8) {
				normal /= normalLength;
				normal = dot(normal, -center) < 0.0 ? -normal : normal;  // face the camera

				float distance = dot(positionCR - center, normal);
				if (distance < ParticleRadius && distance > -CollisionThickness)
					offset = normal * (ParticleRadius - distance);
			}
		}
	}

	return offset;
}

// Pushes a simulation-space position out of the floor plane and out of visible scene surfaces
float3 Collide(float3 position)
{
	position.z = max(position.z, FloorHeight + ParticleRadius);
	[branch] if (DepthCollision != 0)
		position += DepthCollisionOffset(position + OriginRelCam);
	return position;
}

[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSBeginFrame(uint3 id : SV_DispatchThreadID) {
	uint i = id.x;
	if (i >= Capacity)
		return;
	Particle p = Particles[i];
	if (p.Alive == 0)
		return;
	Particles[i].PreviousPosition = p.Position;
}

	[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSEmit(uint3 id : SV_DispatchThreadID)
{
	uint t = id.x;
	if (t >= EmitCount)
		return;

	uint layer = t / DiscCount;
	uint k = t % DiscCount;

	float3 direction = EmitterDirection;
	float3 up = abs(direction.z) < 0.99 ? float3(0, 0, 1) : float3(1, 0, 0);
	float3 tangent = normalize(cross(direction, up));
	float3 bitangent = cross(direction, tangent);

	// Sunflower pattern fills the disc evenly for any particle count
	float discRadius = sqrt((k + 0.5) / DiscCount) * EmitterRadius;
	float angle = k * 2.39996323;  // golden angle
	float3 offset = (tangent * cos(angle) + bitangent * sin(angle)) * discRadius;

	// A small jitter breaks the perfect lattice so the stream doesn't move as rigid sheets
	uint seed = t * 747796405u + FrameIndex * 2891336453u;
	float3 jitter = (float3(FluidSimulation::Random(seed), FluidSimulation::Random(seed + 1), FluidSimulation::Random(seed + 2)) - 0.5) * (0.1 * ParticleRadius);

	float3 velocity = direction * EmitterSpeed;
	float3 position = EmitterPosition + offset + direction * (LayerSpacing * layer) + jitter;

	Particle p;
	p.Position = position;
	p.Age = 0.0;
	p.Velocity = velocity;
	p.Alive = 1;
	p.PreviousPosition = position - velocity * FrameDeltaTime;
	p.pad0 = 0.0;

	Particles[(EmitHead + t) % Capacity] = p;
}

[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSPredict(uint3 id : SV_DispatchThreadID) {
	uint i = id.x;
	if (i >= Capacity)
		return;
	Particle p = ParticlesIn[i];
	if (p.Alive == 0) {
		PredictedOut[i] = float4(p.Position, 0.0);
		return;
	}

	float3 velocity = p.Velocity + Gravity * DeltaTime;
	float3 predicted = p.Position + velocity * DeltaTime;
	PredictedOut[i] = float4(Collide(predicted), 1.0);
}

	[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSClearCells(uint3 id : SV_DispatchThreadID)
{
	if (id.x < FLUID_HASH_TABLE_SIZE)
		CellCounts[id.x] = 0;
}

[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSBuildCells(uint3 id : SV_DispatchThreadID) {
	uint i = id.x;
	if (i >= Capacity || ParticlesIn[i].Alive == 0)
		return;

	uint bucket = FluidSimulation::HashCell(FluidSimulation::GetCell(PredictedIn[i].xyz));
	uint slot;
	InterlockedAdd(CellCounts[bucket], 1, slot);
	// Overfull buckets drop particles from neighbor lookups rather than overflow
	if (slot < FLUID_BUCKET_SIZE)
		CellParticles[bucket * FLUID_BUCKET_SIZE + slot] = i;
}

// Visits every particle within the kernel radius of `position`, excluding `self`.
// The body sees: j (neighbor index), r (position - neighbor), r2 (squared distance).
#define FOR_EACH_NEIGHBOR(position, self, body)                                           \
	{                                                                                     \
		int3 baseCell = FluidSimulation::GetCell(position);                               \
		float h2 = KernelRadius * KernelRadius;                                           \
		[loop] for (int dz = -1; dz <= 1; dz++)                                           \
		{                                                                                 \
			[loop] for (int dy = -1; dy <= 1; dy++)                                       \
			{                                                                             \
				[loop] for (int dx = -1; dx <= 1; dx++)                                   \
				{                                                                         \
					uint bucket = FluidSimulation::HashCell(baseCell + int3(dx, dy, dz)); \
					uint count = min(CellCountsIn[bucket], FLUID_BUCKET_SIZE);            \
					[loop] for (uint n = 0; n < count; n++)                               \
					{                                                                     \
						uint j = CellParticlesIn[bucket * FLUID_BUCKET_SIZE + n];         \
						if (j == self)                                                    \
							continue;                                                     \
						float3 r = position - PredictedIn[j].xyz;                         \
						float r2 = dot(r, r);                                             \
						if (r2 >= h2)                                                     \
							continue;                                                     \
						body                                                              \
					}                                                                     \
				}                                                                         \
			}                                                                             \
		}                                                                                 \
	}

	[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSLambda(uint3 id : SV_DispatchThreadID)
{
	uint i = id.x;
	if (i >= Capacity)
		return;
	if (ParticlesIn[i].Alive == 0) {
		LambdasOut[i] = 0.0;
		return;
	}

	float3 position = PredictedIn[i].xyz;
	float density = FluidSimulation::Poly6(0.0);
	float3 gradientSelf = 0.0;
	float gradientSum = 0.0;

	FOR_EACH_NEIGHBOR(position, i, {
		density += FluidSimulation::Poly6(r2);
		float3 gradient = FluidSimulation::SpikyGrad(r, sqrt(r2)) * InvRestDensity;
		gradientSelf += gradient;
		gradientSum += dot(gradient, gradient);
	})

	// Clamped to positive so the constraint only resists compression; cohesion handles attraction
	float constraint = max(density * InvRestDensity - 1.0, 0.0);
	float lambda = -constraint / (gradientSum + dot(gradientSelf, gradientSelf) + RelaxationEpsilon);
	LambdasOut[i] = float2(lambda, density);
}

[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSDelta(uint3 id : SV_DispatchThreadID) {
	uint i = id.x;
	if (i >= Capacity)
		return;
	float4 predicted = PredictedIn[i];
	if (ParticlesIn[i].Alive == 0) {
		PredictedOut[i] = predicted;
		return;
	}

	float3 position = predicted.xyz;
	float lambdaSelf = LambdasIn[i].x;
	float3 delta = 0.0;

	FOR_EACH_NEIGHBOR(position, i, {
		// Artificial pressure (s_corr) adds surface tension and prevents particle clustering
		float ratio = FluidSimulation::Poly6(r2) * SCorrInvWdq;
		float ratio2 = ratio * ratio;
		float sCorr = -SCorrK * ratio2 * ratio2;
		delta += (lambdaSelf + LambdasIn[j].x + sCorr) * FluidSimulation::SpikyGrad(r, sqrt(r2));
	})

	position += delta * InvRestDensity;
	PredictedOut[i] = float4(Collide(position), 1.0);
}

	[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSVelocity(uint3 id : SV_DispatchThreadID)
{
	uint i = id.x;
	if (i >= Capacity)
		return;
	Particle p = Particles[i];
	if (p.Alive == 0 || DeltaTime <= 0.0) {
		VelocitiesOut[i] = 0.0;
		return;
	}
	VelocitiesOut[i] = float4((PredictedIn[i].xyz - p.Position) / DeltaTime, 0.0);
}

[numthreads(FLUID_GROUP_SIZE, 1, 1)] void CSFinalize(uint3 id : SV_DispatchThreadID) {
	uint i = id.x;
	if (i >= Capacity)
		return;
	Particle p = Particles[i];
	if (p.Alive == 0)
		return;

	float3 position = PredictedIn[i].xyz;
	float3 velocity = VelocitiesIn[i].xyz;

	// XSPH viscosity: blend toward the kernel-weighted average neighbor velocity
	float3 velocityDifference = 0.0;
	float weightSum = 0.0;
	float3 cohesion = 0.0;

	FOR_EACH_NEIGHBOR(position, i, {
		float weight = FluidSimulation::Poly6(r2);
		velocityDifference += (VelocitiesIn[j].xyz - velocity) * weight;
		weightSum += weight;
		float len = sqrt(r2);
		if (len > 1e-6)
			cohesion -= FluidSimulation::Cohesion(len) * (r / len);
	})

	if (weightSum > 0.0)
		velocity += Viscosity * (velocityDifference / weightSum);
	velocity += cohesion * (CohesionAccel * DeltaTime);

	// Guard against solver blow-ups launching particles across the cell
	float speed = length(velocity);
	const float maxSpeed = 3000.0;
	if (speed > maxSpeed)
		velocity *= maxSpeed / speed;

	p.Position = position;
	p.Velocity = velocity;
	p.Age += DeltaTime;
	if (p.Age > Lifetime || position.z < FloorHeight - 1000.0 || any(isnan(position)))
		p.Alive = 0;

	Particles[i] = p;
}
