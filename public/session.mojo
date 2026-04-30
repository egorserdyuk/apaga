from runtime import Session as RuntimeSession, SessionConfig as RuntimeSessionConfig

alias SessionConfig = RuntimeSessionConfig

struct Session:
    var _session: RuntimeSession

    def __init__(out self, model_path: String, config: SessionConfig) raises:
        pass

    fn generate(
        inout self,
        prompt: String,
        max_tokens: Int,
        temperature: Float32,
    ) raises -> String:
        return ""

    fn embed(inout self, text: String) raises:
        pass