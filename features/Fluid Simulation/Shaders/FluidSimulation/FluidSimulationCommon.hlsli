#ifndef __FLUID_SIMULATION_COMMON_HLSLI__
#define __FLUID_SIMULATION_COMMON_HLSLI__

#define FLUID_GROUP_SIZE 256
#define FLUID_HASH_TABLE_SIZE 65536
#define FLUID_BUCKET_SIZE 32

// Must match FluidSimulation::FluidCB
cbuffer FluidCB : register(b0)
{
	float3 OriginRelCam;  // simulation origin relative to the current camera
	float DeltaTime;      // substep length

	float3 OriginRelCamPrev;  // simulation origin relative to the previous frame's camera
	float FrameDeltaTime;

	float3 EmitterPosition;  // in simulation space
	uint EmitHead;

	float3 EmitterDirection;
	uint EmitCount;

	float3 Gravity;
	uint DiscCount;

	float ParticleRadius;
	float KernelRadius;
	float InvRestDensity;
	float RelaxationEpsilon;

	float Poly6Coeff;
	float SpikyGradCoeff;
	float SCorrK;
	float SCorrInvWdq;

	float Viscosity;
	float CohesionAccel;
	float CohesionInvMax;
	float Lifetime;

	float EmitterSpeed;
	float FloorHeight;
	float CollisionThickness;
	uint Capacity;

	uint DepthCollision;
	float LayerSpacing;
	float EmitterRadius;
	uint FrameIndex;

	float4 FluidColor;  // rgb + opacity

	float Roughness;
	float Specular;
	float VelocityStretch;
	float CohesionCoeff;
};

// Must match FluidSimulation::Particle
struct Particle
{
	float3 Position;  // simulation space
	float Age;
	float3 Velocity;
	uint Alive;
	float3 PreviousPosition;  // position at the start of the previous frame, for motion vectors
	float pad0;
};

namespace FluidSimulation
{
	// Poly6 kernel (Mueller 2003), takes squared distance
	float Poly6(float r2)
	{
		float d = max(KernelRadius * KernelRadius - r2, 0.0);
		return Poly6Coeff * d * d * d;
	}

	// Gradient of the spiky kernel with respect to the first particle, for offset r = pi - pj
	float3 SpikyGrad(float3 r, float len)
	{
		float d = max(KernelRadius - len, 0.0);
		return len > 1e-6 ? (SpikyGradCoeff * d * d / len) * r : 0.0;
	}

	// Akinci 2013 cohesion spline, normalized so its peak at h/2 is 1.
	// Negative below roughly h/4, which keeps cohesion from collapsing particles together.
	float Cohesion(float len)
	{
		float h = KernelRadius;
		float a = (h - len);
		float a3r3 = a * a * a * len * len * len;
		float h3 = h * h * h;
		float c = (2.0 * len > h) ? a3r3 : (2.0 * a3r3 - h3 * h3 / 64.0);
		return (len > 0.0 && len < h) ? CohesionCoeff * c * CohesionInvMax : 0.0;
	}

	int3 GetCell(float3 position)
	{
		return int3(floor(position / KernelRadius));
	}

	uint HashCell(int3 cell)
	{
		uint3 c = uint3(cell);
		return ((c.x * 73856093u) ^ (c.y * 19349663u) ^ (c.z * 83492791u)) & (FLUID_HASH_TABLE_SIZE - 1);
	}

	float Random(uint seed)
	{
		seed = (seed ^ 61u) ^ (seed >> 16);
		seed *= 9u;
		seed = seed ^ (seed >> 4);
		seed *= 0x27d4eb2du;
		seed = seed ^ (seed >> 15);
		return float(seed) * (1.0 / 4294967296.0);
	}
}

#endif  // __FLUID_SIMULATION_COMMON_HLSLI__
