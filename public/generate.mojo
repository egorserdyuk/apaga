from runtime import SamplerConfig

struct GenerateConfig:
    var temperature: Float32
    var top_p: Float32
    var top_k: Int
    var max_tokens: Int

    @staticmethod
    fn default() -> GenerateConfig:
        return GenerateConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_tokens=256,
        )

fn generate(
    session,
    prompt: String,
    config: GenerateConfig,
    on_token: fn(String) -> Bool,
) raises:
    pass