cask "mlx-core" do
  version "26.6.9"
  sha256 "31f02096f3031ebf1290ea31c13fbc5204e4bbd2911580ae68c240ad4a6327d0"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLXCore.dmg"
  name "MLX Core"
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  depends_on macos: ">= :sonoma"
  depends_on arch: :arm64

  app "MLX Core.app"

  zap trash: [
    "~/.mlx-serve",
  ]
end
