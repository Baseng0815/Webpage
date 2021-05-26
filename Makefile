SOURCES 	:= $(shell find . -type f -name "*.md")
TARGETS 	:= $(SOURCES:.md=.html)

.PHONY: all install clean

all: $(TARGETS)
	mkdir -p html
	cp $(TARGETS) html
	rm -rf markdown/*.html

%.html: %.md
	pandoc --metadata title="Webpage" -s $< | sed '/<\/style>/a <link rel="stylesheet" href="../css/style.css" type="text/css" media="all"/>' > $@

clean:
	rm -rf html
