OUT_DIR := html

HTML_TARGETS := $(shell find . -type f -name "*.md" -printf "$(OUT_DIR)/%f\n")
HTML_TARGETS := $(HTML_TARGETS:.md=.html)

.PHONY: all install clean

all: $(HTML_TARGETS)
	@echo $(HTML_TARGETS)

$(OUT_DIR)/%.html: markdown/%.md
	mkdir -p $(OUT_DIR)
	pandoc -s $< | sed '/<\/style>/a <link rel="stylesheet" href="../css/style.css" type="text/css" media="all"/>' > $@

clean:
	rm -rf $(HTML_TARGETS)
