# import os
#
# from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
#
# from app.api.deps import verify_api_key
# from app.config import DATA_UPLOAD_DIR
# from app.schemas.upload import UploadResponse
# from app.tools.pandas_tool import load_dataset
# from app.tools.sql_tool import load_dataset_sql
#
# router = APIRouter()
#
# ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")
#
#
# @router.post(
#     "/upload",
#     response_model=UploadResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="Upload a dataset for the agent to analyze",
# )
# def upload_file(file: UploadFile = File(...), _=Depends(verify_api_key)):
#     filename = os.path.basename(file.filename or "")
#     if not filename.lower().endswith(ALLOWED_EXTENSIONS):
#         raise HTTPException(
#             status.HTTP_400_BAD_REQUEST,
#             f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
#         )
#
#     os.makedirs(DATA_UPLOAD_DIR, exist_ok=True)
#     filepath = os.path.join(DATA_UPLOAD_DIR, filename)
#     with open(filepath, "wb") as f:
#         f.write(file.file.read())
#
#     pandas_msg = load_dataset(filepath)
#     sql_msg = load_dataset_sql(filepath)
#     return UploadResponse(status="loaded", filename=filename, detail=f"{pandas_msg} | {sql_msg}")
