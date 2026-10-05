#pragma once

#include "Buffer.h"
#include "Feature.h"

/**
 * @brief GPU particle fluid simulation for small-scale liquids (pouring, streams, pooling).
 *
 * Particles are simulated with Position-Based Fluids (Macklin & Mueller 2013) plus an
 * Akinci-style cohesion term so thin streams hold together. Neighbors are found through a
 * fixed-bucket spatial hash. Particles collide with a floor plane under the emitter and with
 * the scene depth buffer. The simulation and sprite rendering run once per frame right after
 * the deferred composite, so the fluid is lit by the frame's lighting data, depth-tested
 * against the scene and writes motion vectors for TAA and upscalers.
 *
 * Positions are stored relative to a simulation origin captured when the emitter is placed,
 * keeping float precision high far from the worldspace origin.
 */
struct FluidSimulation : Feature
{
	virtual inline std::string GetName() override { return "Fluid Simulation"; }
	virtual std::string GetDisplayName() override { return T("feature.fluid_simulation.name", "Fluid Simulation"); }
	virtual inline std::string GetShortName() override { return "FluidSimulation"; }
	virtual std::string_view GetCategory() const override { return FeatureCategories::kOther; }

	virtual std::pair<std::string, std::vector<std::string>> GetFeatureSummary() override
	{
		return { T("feature.fluid_simulation.description", "Simulates small-scale liquids such as pouring and streams on the GPU, colliding with the scene and pooling on surfaces."),
			{ T("feature.fluid_simulation.key_feature_1", "Position-Based Fluids solver with cohesion for coherent streams"),
				T("feature.fluid_simulation.key_feature_2", "Collision against the scene depth buffer and a floor plane"),
				T("feature.fluid_simulation.key_feature_3", "Adjustable color, transparency, specular and viscosity"),
				T("feature.fluid_simulation.key_feature_4", "Debug emitter for testing in-game") } };
	}

	struct Settings
	{
		bool Enabled = true;
		uint MaxParticlesIndex = 2;  // index into kParticleCapacities
		uint Substeps = 2;
		uint SolverIterations = 3;

		// Particles
		float ParticleRadius = 0.3f;  // game units
		float Lifetime = 8.0f;        // seconds
		float GravityScale = 1.0f;

		// Emitter
		float EmitterSpeed = 150.0f;    // game units per second
		float EmitterRadius = 1.0f;     // game units
		float EmitterPitch = -10.0f;    // degrees, relative to the player's facing
		float EmitterHeight = 100.0f;   // game units above the player's feet
		float EmitterDistance = 60.0f;  // game units in front of the player

		// Solver
		float Cohesion = 0.2f;
		float SurfaceTension = 0.1f;  // PBF artificial pressure (s_corr) strength
		float Relaxation = 0.01f;     // PBF constraint relaxation, as a fraction of the rest gradient sum

		// Collision
		bool DepthCollision = true;
		float CollisionThickness = 20.0f;  // game units behind a visible surface that still count as inside it

		// Fluid appearance and behavior
		float3 Color = { 0.55f, 0.7f, 0.85f };
		float Opacity = 0.6f;
		float Roughness = 0.08f;
		float Specular = 0.02f;  // F0 reflectance
		float Viscosity = 0.05f;
		float VelocityStretch = 0.02f;  // seconds of travel each sprite is stretched along its velocity
	};

	Settings settings;

	static constexpr std::array<uint, 4> kParticleCapacities = { 8192, 16384, 32768, 65536 };

	virtual void SetupResources() override;
	virtual void DrawSettings() override;
	virtual void LoadSettings(json& o_json) override;
	virtual void SaveSettings(json& o_json) override;
	virtual void RestoreDefaultSettings() override;
	virtual void ClearShaderCache() override;

	/**
	 * @brief Steps the simulation and draws the fluid into the main scene target.
	 *
	 * Called from Deferred::DeferredPasses after the deferred composite. Saves and restores
	 * the full D3D11 pipeline state so the game's cached renderer state stays valid.
	 */
	void SimulateAndDraw();

	/** @brief Places the debug emitter in front of the player and starts emitting. */
	void PlaceEmitterAtPlayer();

	/** @brief Stops emitting and removes all particles. */
	void ClearParticles();

	struct alignas(16) Particle
	{
		float3 Position;
		float Age;
		float3 Velocity;
		uint Alive;
		float3 PreviousPosition;
		float pad0;
	};
	STATIC_ASSERT_ALIGNAS_16(Particle);

	struct alignas(16) FluidCB
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

		float4 Color;  // rgb + opacity

		float Roughness;
		float Specular;
		float VelocityStretch;
		float CohesionCoeff;
	};
	STATIC_ASSERT_ALIGNAS_16(FluidCB);

private:
	enum class CS : uint
	{
		BeginFrame,
		Emit,
		Predict,
		ClearCells,
		BuildCells,
		Lambda,
		Delta,
		Velocity,
		Finalize,
		Count
	};

	static constexpr uint kHashTableSize = 1 << 16;
	static constexpr uint kBucketSize = 32;
	static constexpr uint kThreadGroupSize = 256;
	static constexpr uint kMaxDiscParticles = 128;

	ID3D11ComputeShader* GetComputeShader(CS a_shader);
	ID3D11VertexShader* GetVertexShader();
	ID3D11PixelShader* GetPixelShader();

	/** @brief (Re)creates particle and hash buffers sized to the current capacity setting. */
	void CreateParticleBuffers();

	/** @brief Computes kernel coefficients and rest density for the current particle radius. */
	void UpdateSolverConstants(FluidCB& a_cb) const;

	void Simulate(float a_frameDeltaTime);
	void Draw();

	ID3D11ComputeShader* computeShaders[static_cast<uint>(CS::Count)] = {};
	ID3D11VertexShader* vertexShader = nullptr;
	ID3D11PixelShader* pixelShader = nullptr;

	eastl::unique_ptr<ConstantBuffer> fluidCB;
	eastl::unique_ptr<StructuredBuffer> particles;
	eastl::unique_ptr<StructuredBuffer> predictedA;
	eastl::unique_ptr<StructuredBuffer> predictedB;
	eastl::unique_ptr<StructuredBuffer> lambdas;
	eastl::unique_ptr<StructuredBuffer> velocities;
	eastl::unique_ptr<StructuredBuffer> cellCounts;
	eastl::unique_ptr<StructuredBuffer> cellParticles;

	winrt::com_ptr<ID3D11BlendState> blendState;
	winrt::com_ptr<ID3D11DepthStencilState> depthStencilState;
	winrt::com_ptr<ID3D11RasterizerState> rasterizerState;

	uint capacity = 0;
	uint emitHead = 0;
	float emitAccumulator = 0.0f;
	float timeSinceEmit = 0.0f;
	bool emitting = false;
	bool hasParticles = false;
	bool clearRequested = true;

	RE::NiPoint3 simOrigin{};  // world position all particle positions are relative to
	RE::NiPoint3 emitterDirection{ 0.0f, 1.0f, 0.0f };
	float floorHeight = 0.0f;  // simulation-space Z of the floor plane
};
