cask "mlx-core" do
  version "26.4.19"
  sha256 "b9e972b4b83f5f2ffffa1ff9a7f8e1870f88285ca7d19262291acb54ade41f34"

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
