// Debug renderer: each particle is a camera-facing sphere impostor stretched along its
// screen-space velocity. Writes lit color (alpha blended by opacity), depth and motion vectors.

#include "Common/FrameBuffer.hlsli"
#include "Common/MotionBlur.hlsli"
#include "Common/SharedData.hlsli"
#include "FluidSimulation/FluidSimulationCommon.hlsli"

StructuredBuffer<Particle> Particles : register(t0);

struct VSOutput
{
	float4 HPosition : SV_Position;
	float2 Corner : TEXCOORD0;              // [-1, 1] across the sprite
	float3 PositionCR : TEXCOORD1;          // camera-relative
	float3 PreviousPositionCR : TEXCOORD2;  // relative to the previous frame's camera
	float3 AxisX : TEXCOORD3;               // along velocity
	float3 AxisY : TEXCOORD4;
	float3 ToCamera : TEXCOORD5;
};

struct PSOutput
{
	float4 Color : SV_Target0;
	float2 MotionVector : SV_Target1;
};

static const float2 kCorners[6] = {
	float2(-1, -1), float2(1, -1), float2(1, 1),
	float2(-1, -1), float2(1, 1), float2(-1, 1)
};

VSOutput VSMain(uint vertexId : SV_VertexID)
{
	VSOutput output = (VSOutput)0;

	uint index = vertexId / 6;
	Particle p = Particles[index];
	if (p.Alive == 0) {
		output.HPosition = 0.0;  // degenerate, culled
		return output;
	}

	float2 corner = kCorners[vertexId % 6];

	float3 positionCR = p.Position + OriginRelCam;
	float3 previousCR = p.PreviousPosition + OriginRelCamPrev;
	float3 toCamera = normalize(-positionCR);

	// Stretch along the velocity component perpendicular to the view direction
	float3 velocity = p.Velocity - dot(p.Velocity, toCamera) * toCamera;
	float speed = length(velocity);
	float3 axisX;
	if (speed > 1e-3) {
		axisX = velocity / speed;
	} else {
		float3 up = abs(toCamera.z) < 0.99 ? float3(0, 0, 1) : float3(1, 0, 0);
		axisX = normalize(cross(up, toCamera));
	}
	float3 axisY = normalize(cross(toCamera, axisX));

	float radius = ParticleRadius;
	float halfLength = radius + 0.5 * min(speed * VelocityStretch, 4.0 * radius);
	float3 offset = axisX * (corner.x * halfLength) + axisY * (corner.y * radius);

	output.HPosition = mul(FrameBuffer::CameraViewProj, float4(positionCR + offset, 1.0));
	output.Corner = corner;
	output.PositionCR = positionCR + offset;
	output.PreviousPositionCR = previousCR + offset;
	output.AxisX = axisX;
	output.AxisY = axisY;
	output.ToCamera = toCamera;
	return output;
}

float DistributionGGX(float NdotH, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
	return a2 / (3.14159265 * d * d);
}

float VisibilitySmithGGX(float NdotV, float NdotL, float roughness)
{
	float a = roughness * roughness;
	float v = NdotL * sqrt(NdotV * NdotV * (1.0 - a) + a);
	float l = NdotV * sqrt(NdotL * NdotL * (1.0 - a) + a);
	return 0.5 / max(v + l, 1e-5);
}

PSOutput PSMain(VSOutput input)
{
	float2 uv = input.Corner;
	float r2 = dot(uv, uv);
	if (r2 > 1.0)
		discard;

	// Sphere normal across the (possibly stretched) sprite
	float3 normal = normalize(input.AxisX * uv.x + input.AxisY * uv.y + input.ToCamera * sqrt(1.0 - r2));
	float3 view = normalize(input.ToCamera);
	float3 light = SharedData::DirLightDirection.xyz;
	float3 lightColor = SharedData::DirLightColor.rgb;

	float NdotL = saturate(dot(normal, light));
	float NdotV = saturate(dot(normal, view));
	float3 halfVector = normalize(light + view);
	float NdotH = saturate(dot(normal, halfVector));
	float VdotH = saturate(dot(view, halfVector));

	float roughness = max(Roughness, 0.02);
	float fresnelLight = Specular + (1.0 - Specular) * pow(1.0 - VdotH, 5.0);
	float fresnelView = Specular + (1.0 - Specular) * pow(1.0 - NdotV, 5.0);
	float3 specular = DistributionGGX(NdotH, roughness) * VisibilitySmithGGX(NdotV, NdotL, roughness) * fresnelLight * NdotL * lightColor;

	float3 diffuse = FluidColor.rgb * (SharedData::GetAmbient(normal) + lightColor * NdotL);

	// Grazing angles reflect more and transmit less, so opacity rises with view Fresnel.
	// Specular is divided by alpha so blending doesn't dim highlights on clear fluids.
	float alpha = clamp(FluidColor.a + fresnelView * (1.0 - FluidColor.a), 0.05, 1.0);

	PSOutput output;
	output.Color = float4(diffuse + specular / alpha, alpha);
	output.MotionVector = MotionBlur::GetSSMotionVector(float4(input.PositionCR, 1.0), float4(input.PreviousPositionCR, 1.0));
	return output;
}
