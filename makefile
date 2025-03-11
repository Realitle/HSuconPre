all: pipreqs push

install:
	pip install -r requirements.txt --upgrade

pipreqs:
	@if ! command -v pipreqs &> /dev/null; then \
		echo -e "\033[31m请先安装pipreqs: \033[1mpip install pipreqs\033[0m"; exit 1; \
	fi
	@if [ -d ".venv" ]; then \
		source .venv/bin/activate; \
	fi
	pipreqs --force . --mode compat
	sleep 3

push:
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add . && \
		git commit -m "Automated commit by make push" && \
		git push origin main; \
	else \
		echo -e "\033[33m没有需要提交的更改\033[0m"; \
	fi

.PHONY: install clean push