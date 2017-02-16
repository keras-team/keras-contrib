
ifndef DOCKER
	DOCKER=docker
endif

ifndef NV_GPU
	NV_GPU=0
endif
ifndef PYTHON_VERSION
	PYTHON_VERSION=2.7
endif

IMAGE=keras-contrib_py$(PYTHON_VERSION)
BACKEND=tensorflow
NAME=keras_$(PYTHON_VERSION)_$(BACKEND)_$(NV_GPU)
SRC=$(shell pwd)
RUN_ARG=-it -v $(SRC):/src --rm=true --env KERAS_BACKEND=$(BACKEND) --name=$(NAME)

ifeq ($(TEST_MODE), INTEGRATION_TESTS)
	TEST_CMD=py.test tests/integration_tests
else ifeq ($(TEST_MODE), PEP8)
	TEST_CMD=py.test --pep8 -m pep8 -n0
else
	TEST_CMD=py.test tests/ --ignore=tests/integration_tests;
endif


.PHONY: build

all: build

help:
	@cat Makefile

build:
	docker build -t $(IMAGE) --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -f Dockerfile .

run:
	$(DOCKER) run $(RUN_ARG) $(IMAGE) bash -l

test:
	$(DOCKER) run $(RUN_ARG) $(IMAGE) $(TEST_CMD)

testmode:
	echo $(TEST_CMD)

