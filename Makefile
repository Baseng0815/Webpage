OUT_DIR := html

HTML_TARGETS := $(shell find . -type f -name "*.md" -printf "$(OUT_DIR)/%f\n")
HTML_TARGETS := $(HTML_TARGETS:.md=.html)

.PHONY: all install clean

all: $(HTML_TARGETS)

$(OUT_DIR)/%.html: markdown/%.md
	mkdir -p $(OUT_DIR)
	pandoc -s --css "../css/style.css" -f markdown -t html5 -o "$@" "$<"

res: fhd q75

fhd: ./res/gallery/*.JPG
	mogrify -resize 1920x1080^ -gravity Center -extent 1920x1080^ $^

q75: ./res/*.jpg
	mogrify -resize 1080x -quality 75 $^

clean:
	rm -rf $(HTML_TARGETS)
