OUT_DIR := html

HTML_TARGETS := $(shell find ./markdown -type f -name "*.md" -printf "$(OUT_DIR)/%P\n")
HTML_TARGETS := $(HTML_TARGETS:.md=.html)

.PHONY: all install clean

all: $(HTML_TARGETS)

$(OUT_DIR)/%.html: markdown/%.md
	mkdir -p $(@D)
	mkdir -p $(OUT_DIR)
	pandoc -s --css "/css/style.css" -f markdown -t html5 -o "$@" "$<"

res: fhd q70

fhd: ./res/gallery/*.JPG
	mogrify -strip -resize 1920x1080^ -gravity Center -extent 1920x1080^ $^

q70:
	mogrify -strip -resize 720x -quality 70 $(shell find ./res -type f -name *.jpg -size +1M)

clean:
	rm -rf $(HTML_TARGETS)

