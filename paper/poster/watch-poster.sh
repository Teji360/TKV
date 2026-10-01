#!/usr/bin/env bash
# Live preview of the poster in the terminal (Ghostty/kitty graphics protocol).
# Rebuilds and redraws whenever TKV_poster.tex is saved. Ctrl-C to quit.
# Usage: ./watch-poster.sh [dpi]   (default 40; higher = sharper, slower)
set -u
cd "$(dirname "$0")"
TEX=TKV_poster.tex
PNG=$(mktemp -d)/poster
DPI=${1:-40}

show() {
  pdftoppm -png -r "$DPI" -singlefile TKV_poster.pdf "$PNG" || return
  cols=$(tput cols); rows=$(tput lines)
  # Poster is square and terminal cells are ~1:2, so width in cells = 2x height.
  w=$(( (rows - 2) * 2 )); [ "$w" -gt "$cols" ] && w=$cols
  printf '\e_Ga=d\e\\\e[2J\e[H'
  data=$(base64 < "$PNG.png" | tr -d '\n')
  while [ ${#data} -gt 4096 ]; do
    printf '\e_Ga=T,f=100,c=%d,r=%d,q=2,m=1;%s\e\\' "$w" "$((w / 2))" "${data:0:4096}"
    data=${data:4096}
  done
  printf '\e_Gm=0;%s\e\\' "$data"
  printf '\n%s  rebuilt %s  (Ctrl-C to quit)' "$TEX" "$(date +%H:%M:%S)"
}

build() {
  if latexmk -pdf -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null 2>&1; then
    show
  else
    printf '\e_Ga=d\e\\\e[2J\e[H\e[31mBuild failed:\e[0m\n'
    grep -A4 '^!' TKV_poster.log | head -20
  fi
}

trap 'printf "\e_Ga=d\e\\\\\n"; exit' INT
trap show WINCH
last=""
while true; do
  now=$(stat -f %m "$TEX")
  [ "$now" != "$last" ] && { last=$now; build; }
  sleep 1
done
