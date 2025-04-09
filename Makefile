
include .make/base.mk
include .make/python.mk

PROJECT_NAME = ska-sdp-par-model
PROJECT_PATH = ska-telescope/sdp/ska-sdp-config
ARTEFACT_TYPE = python

# Just disable everything we're running afoul of for the moment.
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=E203,E262,E501,E711,E712,E713,E714,E721,E722,E731,F401,F403,F405,F811,F821,F841,W503
PYTHON_SWITCHES_FOR_PYLINT = --disable=invalid-name,unused-wildcard-import,pointless-statement,eval-used,redefined-outer-name,line-too-long,missing-function-docstring,wildcard-import,unused-import,missing-class-docstring,missing-module-docstring,not-callable,unused-variable,consider-using-set-comprehension,broad-exception-raised,consider-using-f-string,use-dict-literal,unnecessary-lambda-assignment,consider-using-generator,too-many-arguments,no-else-return,function-redefined,redefined-builtin,fixme,too-many-lines,superfluous-parens,use-a-generator,dangerous-default-value,useless-object-inheritance,no-member,unidiomatic-typecheck,consider-merging-isinstance,too-many-return-statements,unused-argument,exec-used,raise-missing-from,consider-iterating-dictionary,consider-using-dict-items,unnecessary-pass,too-many-locals,consider-using-in,too-many-statements,too-many-branches,too-few-public-methods,too-many-public-methods,protected-access,inconsistent-return-statements,unreachable,consider-using-max-builtin,ungrouped-imports,too-many-instance-attributes,wrong-import-order,singleton-comparison,undefined-variable,bare-except,astroid-error,unnecessary-lambda,broad-exception-caught
