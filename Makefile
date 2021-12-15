
all: knitr move

knitr:
	echo 'rmarkdown::render("README.Rmd")' | R --no-save

move:
	cp README.html docs/index.html
