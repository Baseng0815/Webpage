SOURCES 	:= $(shell find . -type f -name "*.md")
TARGETS 	:= $(SOURCES:.md=.html)

.PHONY: all install clean

all: $(TARGETS)
	mkdir -p html
	cp $(TARGETS) html
	rm -rf markdown/*.html

%.html: %.md
	cp pre.html $@
	@echo "$(shell markdown $<)" >> $@
	cat post.html >> $@

clean:
	rm -rf html
