{
  description = "copia - Pure Rust rsync-style delta synchronization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rust = pkgs.rust-bin.stable."1.75.0".default.override {
          extensions = [ "rust-src" "rustfmt" "clippy" "llvm-tools-preview" ];
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [ rust pkgs.cargo-llvm-cov pkgs.cargo-mutants ];
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "copia";
          version = "0.1.3";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          buildFeatures = [ "cli" ];
        };
      });
}
