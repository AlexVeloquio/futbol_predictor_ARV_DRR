FROM public.ecr.aws/lambda/python:3.11

COPY src/inference/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/inference/*.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
