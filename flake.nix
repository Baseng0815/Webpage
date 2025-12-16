{
  description = "My Website";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allow-unfree = true;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          pandoc
          gnumake
          imagemagick
        ];
    };
  };
}
