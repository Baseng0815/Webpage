SOURCES 	:= $(shell find . -type f -name "*.md")
TARGETS 	:= $(SOURCES:.md=.html)

.PHONY: all install clean

all: $(TARGETS)
	mkdir -p html
	cp $(TARGETS) html

%.html: %.md
	sed 's/%CONTENT/$(shell markdown $< | sed 's/"/\\"/g' | sed 's/\//\\\//g')/g' template.html > $@

clean:
	rm -rf markdown/*.html html
