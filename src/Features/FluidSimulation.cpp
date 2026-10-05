#include "FluidSimulation.h"

#include "Deferred.h"
#include "Features/Effects11/D3D11StateBackup.h"
#include "Globals.h"
#include "I18n/I18n.h"
#include "State.h"
#include "Utils/D3D.h"
#include "Utils/Game.h"

#include <numbers>

#define I18N_KEY_PREFIX "feature.fluid_simulation."

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
	FluidSimulation::Settings,
	Enabled,
	MaxParticlesIndex,
	Substeps,
	SolverIterations,
	ParticleRadius,
	Lifetime,
	GravityScale,
	EmitterSpeed,
	EmitterRadius,
	EmitterPitch,
	EmitterHeight,
	EmitterDistance,
	Cohesion,
	SurfaceTension,
	Relaxation,
	DepthCollision,
	CollisionThickness,
	Color,
	Opacity,
	Roughness,
	Specular,
	Viscosity,
	VelocityStretch)

namespace
{
	// Skyrim units: 1 unit ~= 1.428 cm, so 9.81 m/s^2 ~= 686 units/s^2
	constexpr float kGravity = 686.0f;
	// Longest frame the simulation will step; longer frames (hitches, loading) are clamped.
	constexpr float kMaxFrameDeltaTime = 1.0f / 20.0f;

	constexpr const char* kComputeEntryPoints[] = {
		"CSBeginFrame",
		"CSEmit",
		"CSPredict",
		"CSClearCells",
		"CSBuildCells",
		"CSLambda",
		"CSDelta",
		"CSVelocity",
		"CSFinalize",
	};
	static_assert(std::size(kComputeEntryPoints) == 9);

	uint DispatchGroups(uint a_count, uint a_groupSize)
	{
		return (a_count + a_groupSize - 1) / a_groupSize;
	}
}

void FluidSimulation::DrawSettings()
{
	ImGui::Checkbox(T(TKEY("enabled"), "Enabled"), &settings.Enabled);

	if (ImGui::TreeNodeEx(T(TKEY("debug_emitter"), "Debug Emitter"), ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(T(TKEY("place_emitter"), "Place Emitter In Front Of Player")))
			PlaceEmitterAtPlayer();
		ImGui::SameLine();
		if (emitting) {
			if (ImGui::Button(T(TKEY("stop_emitter"), "Stop Emitting")))
				emitting = false;
		} else if (hasParticles) {
			if (ImGui::Button(T(TKEY("resume_emitter"), "Resume Emitting")))
				emitting = true;
		}
		ImGui::SameLine();
		if (ImGui::Button(T(TKEY("clear_particles"), "Clear Particles")))
			ClearParticles();

		ImGui::SliderFloat(T(TKEY("emitter_speed"), "Speed"), &settings.EmitterSpeed, 10.0f, 600.0f, "%.0f");
		ImGui::SliderFloat(T(TKEY("emitter_radius"), "Stream Radius"), &settings.EmitterRadius, 0.2f, 5.0f, "%.2f");
		ImGui::SliderFloat(T(TKEY("emitter_pitch"), "Pitch"), &settings.EmitterPitch, -90.0f, 60.0f, "%.0f");
		ImGui::SliderFloat(T(TKEY("emitter_height"), "Height"), &settings.EmitterHeight, 0.0f, 250.0f, "%.0f");
		ImGui::SliderFloat(T(TKEY("emitter_distance"), "Distance"), &settings.EmitterDistance, 10.0f, 300.0f, "%.0f");
		if (auto _tt = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("emitter_placement_tooltip"), "Height and distance apply the next time the emitter is placed."));
		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx(T(TKEY("fluid"), "Fluid"), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3(T(TKEY("color"), "Color"), &settings.Color.x);
		ImGui::SliderFloat(T(TKEY("opacity"), "Opacity"), &settings.Opacity, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T(TKEY("roughness"), "Roughness"), &settings.Roughness, 0.02f, 1.0f, "%.2f");
		ImGui::SliderFloat(T(TKEY("specular"), "Specular"), &settings.Specular, 0.0f, 0.2f, "%.3f");
		ImGui::SliderFloat(T(TKEY("viscosity"), "Viscosity"), &settings.Viscosity, 0.0f, 1.0f, "%.2f");
		if (auto _tt = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("viscosity_tooltip"), "Low values splash and break up; high values flow slowly and hold together."));
		ImGui::TreePop();
	}

	if (ImGui::TreeNodeEx(T(TKEY("simulation"), "Simulation"))) {
		int capacityIndex = static_cast<int>(settings.MaxParticlesIndex);
		const char* capacityLabels[] = { "8192", "16384", "32768", "65536" };
		if (ImGui::Combo(T(TKEY("max_particles"), "Max Particles"), &capacityIndex, capacityLabels, static_cast<int>(std::size(capacityLabels))))
			settings.MaxParticlesIndex = static_cast<uint>(capacityIndex);

		ImGui::SliderFloat(T(TKEY("particle_radius"), "Particle Radius"), &settings.ParticleRadius, 0.1f, 2.0f, "%.2f");
		ImGui::SliderFloat(T(TKEY("lifetime"), "Lifetime"), &settings.Lifetime, 1.0f, 30.0f, "%.1f s");
		ImGui::SliderFloat(T(TKEY("gravity_scale"), "Gravity Scale"), &settings.GravityScale, 0.0f, 2.0f, "%.2f");

		int substeps = static_cast<int>(settings.Substeps);
		if (ImGui::SliderInt(T(TKEY("substeps"), "Substeps"), &substeps, 1, 6))
			settings.Substeps = static_cast<uint>(substeps);
		int iterations = static_cast<int>(settings.SolverIterations);
		if (ImGui::SliderInt(T(TKEY("solver_iterations"), "Solver Iterations"), &iterations, 1, 8))
			settings.SolverIterations = static_cast<uint>(iterations);

		ImGui::SliderFloat(T(TKEY("cohesion"), "Cohesion"), &settings.Cohesion, 0.0f, 2.0f, "%.2f");
		if (auto _tt = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("cohesion_tooltip"), "How strongly particles pull together, keeping streams and pools in one body."));
		ImGui::SliderFloat(T(TKEY("surface_tension"), "Surface Tension"), &settings.SurfaceTension, 0.0f, 0.5f, "%.3f");
		ImGui::SliderFloat(T(TKEY("relaxation"), "Relaxation"), &settings.Relaxation, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		if (auto _tt = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("relaxation_tooltip"), "Softens the incompressibility constraint. Raise it if the fluid jitters or explodes."));
		ImGui::SliderFloat(T(TKEY("velocity_stretch"), "Velocity Stretch"), &settings.VelocityStretch, 0.0f, 0.1f, "%.3f s");

		ImGui::Checkbox(T(TKEY("depth_collision"), "Depth Buffer Collision"), &settings.DepthCollision);
		ImGui::SliderFloat(T(TKEY("collision_thickness"), "Collision Thickness"), &settings.CollisionThickness, 1.0f, 100.0f, "%.0f");
		if (auto _tt = Util::HoverTooltipWrapper())
			ImGui::Text("%s", T(TKEY("collision_thickness_tooltip"), "How far behind a visible surface a particle still counts as inside it."));
		ImGui::TreePop();
	}

	ImGui::Text("%s", std::format("{}: {}", T(TKEY("capacity"), "Particle Capacity"), capacity).c_str());
}

void FluidSimulation::LoadSettings(json& o_json)
{
	settings = o_json;
	settings.MaxParticlesIndex = std::min<uint>(settings.MaxParticlesIndex, static_cast<uint>(kParticleCapacities.size() - 1));
	settings.Substeps = std::clamp<uint>(settings.Substeps, 1, 6);
	settings.SolverIterations = std::clamp<uint>(settings.SolverIterations, 1, 8);
	settings.ParticleRadius = std::clamp(settings.ParticleRadius, 0.1f, 2.0f);
}

void FluidSimulation::SaveSettings(json& o_json)
{
	o_json = settings;
}

void FluidSimulation::RestoreDefaultSettings()
{
	settings = {};
}

void FluidSimulation::SetupResources()
{
	auto device = globals::d3d::device;

	fluidCB = eastl::make_unique<ConstantBuffer>(ConstantBufferDesc<FluidCB>(), "FluidSimulation::FluidCB");

	CreateParticleBuffers();

	{
		D3D11_BLEND_DESC desc{};
		desc.IndependentBlendEnable = TRUE;
		// Main color: alpha blended by fluid opacity
		desc.RenderTarget[0].BlendEnable = TRUE;
		desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
		desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
		desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
		desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
		desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
		desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		// Motion vectors: overwritten so TAA reprojects the fluid, not what is behind it
		desc.RenderTarget[1].BlendEnable = FALSE;
		desc.RenderTarget[1].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN;
		DX::ThrowIfFailed(device->CreateBlendState(&desc, blendState.put()));
		Util::SetResourceName(blendState.get(), "FluidSimulation::BlendState");
	}

	{
		D3D11_DEPTH_STENCIL_DESC desc{};
		desc.DepthEnable = TRUE;
		desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
		desc.StencilEnable = FALSE;
		DX::ThrowIfFailed(device->CreateDepthStencilState(&desc, depthStencilState.put()));
		Util::SetResourceName(depthStencilState.get(), "FluidSimulation::DepthStencilState");
	}

	{
		D3D11_RASTERIZER_DESC desc{};
		desc.FillMode = D3D11_FILL_SOLID;
		desc.CullMode = D3D11_CULL_NONE;
		desc.DepthClipEnable = TRUE;
		DX::ThrowIfFailed(device->CreateRasterizerState(&desc, rasterizerState.put()));
		Util::SetResourceName(rasterizerState.get(), "FluidSimulation::RasterizerState");
	}
}

void FluidSimulation::CreateParticleBuffers()
{
	capacity = kParticleCapacities[settings.MaxParticlesIndex];

	auto makeBuffer = [](auto a_tag, uint a_count, const char* a_name) {
		using T = decltype(a_tag);
		auto buffer = eastl::make_unique<StructuredBuffer>(StructuredBufferDesc<T>(a_count, true, false), a_count, a_name);
		buffer->CreateSRV();
		buffer->CreateUAV();
		return buffer;
	};

	particles = makeBuffer(Particle{}, capacity, "FluidSimulation::Particles");
	predictedA = makeBuffer(float4{}, capacity, "FluidSimulation::PredictedA");
	predictedB = makeBuffer(float4{}, capacity, "FluidSimulation::PredictedB");
	lambdas = makeBuffer(float2{}, capacity, "FluidSimulation::Lambdas");
	velocities = makeBuffer(float4{}, capacity, "FluidSimulation::Velocities");
	cellCounts = makeBuffer(uint{}, kHashTableSize, "FluidSimulation::CellCounts");
	cellParticles = makeBuffer(uint{}, kHashTableSize * kBucketSize, "FluidSimulation::CellParticles");

	emitHead = 0;
	emitAccumulator = 0.0f;
	hasParticles = false;
	clearRequested = true;
}

void FluidSimulation::ClearShaderCache()
{
	for (auto& shader : computeShaders) {
		if (shader)
			shader->Release();
		shader = nullptr;
	}
	if (vertexShader)
		vertexShader->Release();
	vertexShader = nullptr;
	if (pixelShader)
		pixelShader->Release();
	pixelShader = nullptr;
}

ID3D11ComputeShader* FluidSimulation::GetComputeShader(CS a_shader)
{
	auto& shader = computeShaders[static_cast<uint>(a_shader)];
	if (!shader) {
		const char* entryPoint = kComputeEntryPoints[static_cast<uint>(a_shader)];
		logger::debug("[Fluid Simulation] Compiling {}", entryPoint);
		shader = static_cast<ID3D11ComputeShader*>(Util::CompileShader(L"Data\\Shaders\\FluidSimulation\\FluidSimulationCS.hlsl", {}, "cs_5_0", entryPoint));
	}
	return shader;
}

ID3D11VertexShader* FluidSimulation::GetVertexShader()
{
	if (!vertexShader) {
		logger::debug("[Fluid Simulation] Compiling FluidRender VS");
		vertexShader = static_cast<ID3D11VertexShader*>(Util::CompileShader(L"Data\\Shaders\\FluidSimulation\\FluidRender.hlsl", {}, "vs_5_0", "VSMain"));
	}
	return vertexShader;
}

ID3D11PixelShader* FluidSimulation::GetPixelShader()
{
	if (!pixelShader) {
		logger::debug("[Fluid Simulation] Compiling FluidRender PS");
		pixelShader = static_cast<ID3D11PixelShader*>(Util::CompileShader(L"Data\\Shaders\\FluidSimulation\\FluidRender.hlsl", {}, "ps_5_0", "PSMain"));
	}
	return pixelShader;
}

void FluidSimulation::PlaceEmitterAtPlayer()
{
	auto player = RE::PlayerCharacter::GetSingleton();
	if (!player || !player->Is3DLoaded())
		return;

	const RE::NiPoint3 feet = player->GetPosition();
	const float heading = player->GetAngleZ();
	const RE::NiPoint3 forward{ std::sin(heading), std::cos(heading), 0.0f };

	// Re-anchor the simulation at the emitter; existing particles are discarded because
	// their positions are relative to the old origin.
	simOrigin = feet + forward * settings.EmitterDistance + RE::NiPoint3{ 0.0f, 0.0f, settings.EmitterHeight };
	emitterDirection = forward;
	floorHeight = feet.z - simOrigin.z;

	clearRequested = true;
	hasParticles = false;
	emitting = true;
	emitAccumulator = 0.0f;
}

void FluidSimulation::ClearParticles()
{
	emitting = false;
	hasParticles = false;
	clearRequested = true;
}

void FluidSimulation::UpdateSolverConstants(FluidCB& a_cb) const
{
	const float radius = settings.ParticleRadius;
	const float h = 4.0f * radius;  // kernel support radius
	const float spacing = 2.0f * radius;
	const float pi = std::numbers::pi_v<float>;

	const float h2 = h * h;
	const float h6 = h2 * h2 * h2;
	const float h9 = h6 * h2 * h;
	const float poly6Coeff = 315.0f / (64.0f * pi * h9);
	const float spikyGradCoeff = -45.0f / (pi * h6);

	auto poly6 = [&](float r2) { return r2 < h2 ? poly6Coeff * std::pow(h2 - r2, 3.0f) : 0.0f; };
	auto spikyGrad = [&](float r) { return (r > 0.0f && r < h) ? spikyGradCoeff * (h - r) * (h - r) : 0.0f; };

	// Rest density and constraint gradient sum of a particle inside a cubic lattice at rest
	// spacing. Mass is 1, so density is the kernel sum including the particle itself.
	float restDensity = 0.0f;
	float gradSum = 0.0f;
	const int extent = static_cast<int>(std::ceil(h / spacing));
	for (int x = -extent; x <= extent; x++) {
		for (int y = -extent; y <= extent; y++) {
			for (int z = -extent; z <= extent; z++) {
				const float r2 = (float(x * x + y * y + z * z)) * spacing * spacing;
				restDensity += poly6(r2);
				const float g = spikyGrad(std::sqrt(r2));
				gradSum += g * g;
			}
		}
	}
	restDensity = std::max(restDensity, 1e-6f);
	const float invRestDensity = 1.0f / restDensity;
	gradSum *= invRestDensity * invRestDensity;

	const float deltaQ = 0.2f * h;
	const float wDeltaQ = poly6(deltaQ * deltaQ);

	a_cb.ParticleRadius = radius;
	a_cb.KernelRadius = h;
	a_cb.InvRestDensity = invRestDensity;
	a_cb.RelaxationEpsilon = std::max(settings.Relaxation * gradSum, 1e-12f);
	a_cb.Poly6Coeff = poly6Coeff;
	a_cb.SpikyGradCoeff = spikyGradCoeff;
	a_cb.SCorrK = settings.SurfaceTension;
	a_cb.SCorrInvWdq = wDeltaQ > 0.0f ? 1.0f / wDeltaQ : 0.0f;
	// Akinci 2013 cohesion spline, normalized so its peak (r = h/2) is 1
	a_cb.CohesionCoeff = 32.0f / (pi * h9);
	a_cb.CohesionInvMax = 2.0f * pi * h2 * h;
	a_cb.CohesionAccel = settings.Cohesion * kGravity;
	a_cb.LayerSpacing = spacing;
}

void FluidSimulation::SimulateAndDraw()
{
	if (!settings.Enabled || !fluidCB)
		return;

	// Max Particles changed in the menu; existing particles are discarded with the old buffers
	if (capacity != kParticleCapacities[settings.MaxParticlesIndex])
		CreateParticleBuffers();

	if (!emitting && !hasParticles && !clearRequested)
		return;

	ZoneScoped;
	TracyD3D11Zone(globals::state->tracyCtx, "FluidSimulation");

	auto context = globals::d3d::context;

	Effects11Util::D3D11FullStateBackup stateBackup;
	stateBackup.Save(context);

	const float frameDeltaTime = std::clamp(*globals::game::deltaTime, 0.0f, kMaxFrameDeltaTime);
	Simulate(frameDeltaTime);
	if (hasParticles)
		Draw();

	stateBackup.Restore(context);
}

void FluidSimulation::Simulate(float a_frameDeltaTime)
{
	auto context = globals::d3d::context;
	auto shadowState = globals::game::shadowState;

	const RE::NiPoint3 eye = shadowState->GetRuntimeData().posAdjust.getEye();
	const RE::NiPoint3 previousEye = shadowState->GetRuntimeData().previousPosAdjust.getEye();

	FluidCB cb{};
	UpdateSolverConstants(cb);

	const RE::NiPoint3 originRelCam = simOrigin - eye;
	const RE::NiPoint3 originRelCamPrev = simOrigin - previousEye;
	cb.OriginRelCam = { originRelCam.x, originRelCam.y, originRelCam.z };
	cb.OriginRelCamPrev = { originRelCamPrev.x, originRelCamPrev.y, originRelCamPrev.z };
	cb.FrameDeltaTime = a_frameDeltaTime;
	cb.Gravity = { 0.0f, 0.0f, -kGravity * settings.GravityScale };
	cb.Viscosity = settings.Viscosity;
	cb.Lifetime = settings.Lifetime;
	cb.FloorHeight = floorHeight;
	cb.CollisionThickness = settings.CollisionThickness;
	cb.Capacity = capacity;
	cb.DepthCollision = settings.DepthCollision ? 1u : 0u;
	cb.FrameIndex = globals::state->frameCount;
	cb.Color = { settings.Color.x, settings.Color.y, settings.Color.z, settings.Opacity };
	cb.Roughness = settings.Roughness;
	cb.Specular = settings.Specular;
	cb.VelocityStretch = settings.VelocityStretch;

	// Emitter: a disc of particles, emitted in layers one particle spacing apart so the
	// stream starts evenly packed instead of clumping
	const float pitch = settings.EmitterPitch * std::numbers::pi_v<float> / 180.0f;
	RE::NiPoint3 direction = emitterDirection * std::cos(pitch) + RE::NiPoint3{ 0.0f, 0.0f, std::sin(pitch) };
	direction.Unitize();
	cb.EmitterPosition = { 0.0f, 0.0f, 0.0f };
	cb.EmitterDirection = { direction.x, direction.y, direction.z };
	cb.EmitterSpeed = settings.EmitterSpeed;
	cb.EmitterRadius = settings.EmitterRadius;

	const float discArea = std::numbers::pi_v<float> * settings.EmitterRadius * settings.EmitterRadius;
	const uint discCount = std::clamp(static_cast<uint>(std::round(discArea / (cb.LayerSpacing * cb.LayerSpacing))), 1u, kMaxDiscParticles);
	cb.DiscCount = discCount;

	uint emitCount = 0;
	if (emitting && a_frameDeltaTime > 0.0f) {
		emitAccumulator += a_frameDeltaTime * settings.EmitterSpeed / cb.LayerSpacing;
		const uint maxLayers = std::max(1u, capacity / (4u * discCount));
		const uint layers = std::min(static_cast<uint>(emitAccumulator), maxLayers);
		emitAccumulator -= static_cast<float>(layers);
		emitAccumulator = std::min(emitAccumulator, 1.0f);
		emitCount = layers * discCount;
	}
	cb.EmitHead = emitHead;
	cb.EmitCount = emitCount;

	const uint substeps = std::max(settings.Substeps, 1u);
	cb.DeltaTime = a_frameDeltaTime / static_cast<float>(substeps);

	fluidCB->Update(cb);

	ID3D11Buffer* cbs[] = { fluidCB->CB() };
	context->CSSetConstantBuffers(0, 1, cbs);
	ID3D11Buffer* sharedCBs[] = { globals::state->sharedDataCB->CB() };
	context->CSSetConstantBuffers(5, 1, sharedCBs);
	ID3D11Buffer* perFrameCBs[] = { *globals::game::perFrame };
	context->CSSetConstantBuffers(12, 1, perFrameCBs);

	ID3D11ShaderResourceView* depthSRV = Util::GetCurrentSceneDepthSRV(false);

	const uint particleGroups = DispatchGroups(capacity, kThreadGroupSize);

	// Every pass shares one register layout (see FluidSimulationCS.hlsl); unused slots stay null
	struct Bindings
	{
		ID3D11ShaderResourceView* particles = nullptr;          // t0
		ID3D11ShaderResourceView* predicted = nullptr;          // t1
		ID3D11ShaderResourceView* cellCounts = nullptr;         // t2
		ID3D11ShaderResourceView* cellParticles = nullptr;      // t3
		ID3D11ShaderResourceView* lambdas = nullptr;            // t4
		ID3D11ShaderResourceView* velocities = nullptr;         // t5
		ID3D11ShaderResourceView* depth = nullptr;              // t6
		ID3D11UnorderedAccessView* particlesOut = nullptr;      // u0
		ID3D11UnorderedAccessView* predictedOut = nullptr;      // u1
		ID3D11UnorderedAccessView* cellCountsOut = nullptr;     // u2
		ID3D11UnorderedAccessView* cellParticlesOut = nullptr;  // u3
		ID3D11UnorderedAccessView* lambdasOut = nullptr;        // u4
		ID3D11UnorderedAccessView* velocitiesOut = nullptr;     // u5
	};

	auto dispatch = [&](CS a_shader, const Bindings& a_bindings, uint a_groups) {
		auto shader = GetComputeShader(a_shader);
		if (!shader)
			return;
		ID3D11ShaderResourceView* srvs[7] = { a_bindings.particles, a_bindings.predicted, a_bindings.cellCounts, a_bindings.cellParticles, a_bindings.lambdas, a_bindings.velocities, a_bindings.depth };
		ID3D11UnorderedAccessView* uavs[6] = { a_bindings.particlesOut, a_bindings.predictedOut, a_bindings.cellCountsOut, a_bindings.cellParticlesOut, a_bindings.lambdasOut, a_bindings.velocitiesOut };
		context->CSSetShaderResources(0, ARRAYSIZE(srvs), srvs);
		context->CSSetUnorderedAccessViews(0, ARRAYSIZE(uavs), uavs, nullptr);
		context->CSSetShader(shader, nullptr, 0);
		context->Dispatch(a_groups, 1, 1);

		ID3D11ShaderResourceView* nullSRVs[7] = {};
		ID3D11UnorderedAccessView* nullUAVs[6] = {};
		context->CSSetShaderResources(0, ARRAYSIZE(nullSRVs), nullSRVs);
		context->CSSetUnorderedAccessViews(0, ARRAYSIZE(nullUAVs), nullUAVs, nullptr);
	};

	globals::profiler->BeginPass("FluidSimulation::Simulate");

	if (clearRequested) {
		const UINT zero[4] = { 0, 0, 0, 0 };
		context->ClearUnorderedAccessViewUint(particles->UAV(), zero);
		emitHead = 0;
		clearRequested = false;
	}

	// Frame start: remember positions for motion vectors, then spawn new particles
	dispatch(CS::BeginFrame, { .particlesOut = particles->UAV() }, particleGroups);
	if (emitCount > 0) {
		dispatch(CS::Emit, { .particlesOut = particles->UAV() }, DispatchGroups(emitCount, kThreadGroupSize));
		emitHead = (emitHead + emitCount) % capacity;
		hasParticles = true;
		timeSinceEmit = 0.0f;
	} else if (hasParticles) {
		// Every particle has outlived its lifetime once this much time passes without
		// emission, so the simulation can go idle
		timeSinceEmit += a_frameDeltaTime;
		if (timeSinceEmit > settings.Lifetime)
			hasParticles = false;
	}

	if (a_frameDeltaTime > 0.0f) {
		for (uint step = 0; step < substeps; step++) {
			dispatch(CS::Predict, { .particles = particles->SRV(), .depth = depthSRV, .predictedOut = predictedA->UAV() }, particleGroups);

			dispatch(CS::ClearCells, { .cellCountsOut = cellCounts->UAV() }, DispatchGroups(kHashTableSize, kThreadGroupSize));
			dispatch(CS::BuildCells, { .particles = particles->SRV(), .predicted = predictedA->SRV(), .cellCountsOut = cellCounts->UAV(), .cellParticlesOut = cellParticles->UAV() }, particleGroups);

			// Jacobi iterations ping-pong between the two predicted-position buffers
			StructuredBuffer* src = predictedA.get();
			StructuredBuffer* dst = predictedB.get();
			for (uint iteration = 0; iteration < settings.SolverIterations; iteration++) {
				dispatch(CS::Lambda, { .particles = particles->SRV(), .predicted = src->SRV(), .cellCounts = cellCounts->SRV(), .cellParticles = cellParticles->SRV(), .lambdasOut = lambdas->UAV() }, particleGroups);
				dispatch(CS::Delta, { .particles = particles->SRV(), .predicted = src->SRV(), .cellCounts = cellCounts->SRV(), .cellParticles = cellParticles->SRV(), .lambdas = lambdas->SRV(), .depth = depthSRV, .predictedOut = dst->UAV() }, particleGroups);
				std::swap(src, dst);
			}

			dispatch(CS::Velocity, { .predicted = src->SRV(), .particlesOut = particles->UAV(), .velocitiesOut = velocities->UAV() }, particleGroups);
			dispatch(CS::Finalize, { .predicted = src->SRV(), .cellCounts = cellCounts->SRV(), .cellParticles = cellParticles->SRV(), .velocities = velocities->SRV(), .particlesOut = particles->UAV() }, particleGroups);
		}
	}

	globals::profiler->EndPass();

	ID3D11Buffer* nullCB[1] = {};
	context->CSSetConstantBuffers(0, 1, nullCB);
	context->CSSetShader(nullptr, nullptr, 0);
}

void FluidSimulation::Draw()
{
	auto context = globals::d3d::context;
	auto renderer = globals::game::renderer;
	auto vs = GetVertexShader();
	auto ps = GetPixelShader();
	if (!vs || !ps)
		return;

	globals::profiler->BeginPass("FluidSimulation::Draw");

	auto& main = renderer->GetRuntimeData().renderTargets[Deferred::GetSingleton()->forwardRenderTargets[0]];
	auto& motionVectors = renderer->GetRuntimeData().renderTargets[RE::RENDER_TARGETS::kMOTION_VECTOR];
	auto& depth = renderer->GetDepthStencilData().depthStencils[RE::RENDER_TARGETS_DEPTHSTENCIL::kMAIN];

	ID3D11RenderTargetView* rtvs[2] = { main.RTV, motionVectors.RTV };
	context->OMSetRenderTargets(2, rtvs, depth.views[0]);
	context->OMSetBlendState(blendState.get(), nullptr, 0xFFFFFFFF);
	context->OMSetDepthStencilState(depthStencilState.get(), 0);
	context->RSSetState(rasterizerState.get());

	const auto& graphicsState = globals::game::graphicsState;
	const float2 resolution = Util::ConvertToDynamic(float2{ static_cast<float>(graphicsState->screenWidth), static_cast<float>(graphicsState->screenHeight) });
	D3D11_VIEWPORT viewport{ 0.0f, 0.0f, resolution.x, resolution.y, 0.0f, 1.0f };
	context->RSSetViewports(1, &viewport);

	context->IASetInputLayout(nullptr);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	ID3D11Buffer* nullVB = nullptr;
	UINT zero = 0;
	context->IASetVertexBuffers(0, 1, &nullVB, &zero, &zero);

	ID3D11Buffer* cbs[] = { fluidCB->CB() };
	ID3D11Buffer* sharedCBs[] = { globals::state->sharedDataCB->CB() };
	ID3D11Buffer* perFrameCBs[] = { *globals::game::perFrame };
	context->VSSetConstantBuffers(0, 1, cbs);
	context->VSSetConstantBuffers(5, 1, sharedCBs);
	context->VSSetConstantBuffers(12, 1, perFrameCBs);
	context->PSSetConstantBuffers(0, 1, cbs);
	context->PSSetConstantBuffers(5, 1, sharedCBs);
	context->PSSetConstantBuffers(12, 1, perFrameCBs);

	ID3D11ShaderResourceView* srvs[] = { particles->SRV() };
	context->VSSetShaderResources(0, 1, srvs);

	// The full state backup does not cover the geometry shader stage, so restore it by hand
	winrt::com_ptr<ID3D11GeometryShader> previousGS;
	context->GSGetShader(previousGS.put(), nullptr, nullptr);

	context->VSSetShader(vs, nullptr, 0);
	context->GSSetShader(nullptr, nullptr, 0);
	context->PSSetShader(ps, nullptr, 0);

	context->Draw(capacity * 6, 0);

	context->GSSetShader(previousGS.get(), nullptr, 0);

	ID3D11ShaderResourceView* nullSRV[1] = {};
	context->VSSetShaderResources(0, 1, nullSRV);

	globals::profiler->EndPass();
}

#undef I18N_KEY_PREFIX
