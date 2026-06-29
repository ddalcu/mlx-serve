class MlxServe < Formula
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"
  version "26.6.13"
  sha256 "2af6cc95fd8354b7e24acaabedd2c4e3ea402f396fde99b3966af439c3056ee8"
  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/mlx-serve-bin-macos-arm64.tar.gz"

  depends_on macos: :sonoma
  depends_on arch: :arm64

  def install
    libexec.install Dir["lib/*"]
    bin.install "mlx-serve"

    # Fix rpaths to use bundled libs in libexec (avoids conflicts with mlx/mlx-c)
    system "install_name_tool", "-change",
           "@executable_path/lib/libmlxc.dylib",
           "#{libexec}/libmlxc.dylib",
           "#{bin}/mlx-serve"

    system "install_name_tool", "-change",
           "@loader_path/libmlx.dylib",
           "#{libexec}/libmlx.dylib",
           "#{libexec}/libmlxc.dylib"

    # libwebp ships bundled with install_name @executable_path/lib/libwebp.dylib;
    # repoint to libexec since Homebrew splits bin/ and libexec/.
    if File.exist?("#{libexec}/libwebp.dylib")
      system "install_name_tool", "-change",
             "@executable_path/lib/libwebp.dylib",
             "#{libexec}/libwebp.dylib",
             "#{bin}/mlx-serve"
    end

    if File.exist?("#{libexec}/libllama.dylib")
      system "install_name_tool", "-change",
             "@executable_path/lib/libllama.dylib",
             "#{libexec}/libllama.dylib",
             "#{bin}/mlx-serve"
    end

    # Re-sign after rpath patching (hardened runtime invalidates on modification)
    system "codesign", "--force", "--sign", "-", "#{libexec}/libmlxc.dylib"
    system "codesign", "--force", "--sign", "-", "#{bin}/mlx-serve"
  end

  test do
    assert_match "mlx-serve", shell_output("#{bin}/mlx-serve --version 2>&1", 0)
    assert_match "Usage", shell_output("#{bin}/mlx-serve --help 2>&1", 0)
  end
end
