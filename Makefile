OUT_DIR := html

HTML_TARGETS := $(shell find ./markdown -type f -name "*.md")
HTML_TARGETS := $(subst ./markdown/, ./html/, $(HTML_TARGETS))
HTML_TARGETS := $(HTML_TARGETS:.md=.html)

.PHONY: all install clean q70

all: $(HTML_TARGETS) q70

$(OUT_DIR)/%.html: markdown/%.md
	mkdir -p $(@D)
	mkdir -p $(OUT_DIR)
	pandoc --mathjax -s --css "/css/style.css" -f markdown -t html5 -o "$@" "$<"

q70:
	mogrify -strip -resize 720x -quality 70 $(shell find ./res -path ./res/hq -prune -o -type f -name "*.jpg" -size +1M -print)

clean:
	rm -rf $(HTML_TARGETS)

