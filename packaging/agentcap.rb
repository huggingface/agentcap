class Agentcap < Formula
  include Language::Python::Virtualenv

  desc "Run coding agents at scale, capture every chat-completion byte, publish to the Hub"
  homepage "https://github.com/huggingface/agentcap"
  url "https://github.com/huggingface/agentcap/archive/refs/tags/v0.0.1.tar.gz"
  sha256 "TODO_SET_AFTER_FIRST_TAG"
  license "Apache-2.0"

  depends_on "python@3.12"
  depends_on "fzf"
  depends_on "trufflehog"

  on_macos do
    depends_on "lima"
  end

  on_linux do
    depends_on "bubblewrap"
    depends_on "buildah"
  end

  # Transitive Python deps. Regenerate after every pyproject.toml change:
  #   brew update-python-resources packaging/agentcap.rb
  # (resource blocks intentionally empty for now — populate before first
  # real install.)

  def install
    virtualenv_install_with_resources
  end

  def caveats
    return unless OS.linux?
    <<~EOS
      On Ubuntu 24.04+ the bwrap sandbox needs unprivileged user
      namespaces. Run once:

        sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
        echo 'kernel.apparmor_restrict_unprivileged_userns=0' \\
          | sudo tee /etc/sysctl.d/60-agentcap-bwrap.conf

      Homebrew can't do this — root owns the sysctl.
    EOS
  end

  test do
    system bin/"agentcap", "--help"
  end
end
