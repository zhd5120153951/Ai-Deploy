import os
from flask import Blueprint, request, render_template, jsonify, current_app

from applications.common.utils.http import fail_api, success_api, table_api
from applications.common.utils.rights import authorize
from applications.extensions import db
from applications.models import Photo
from applications.common.utils import upload as upload_curd

bp = Blueprint('cameraadd', __name__, url_prefix='/cameraadd')


#  图片上传管理界面
@bp.get('/')
@authorize("system:cameraadd:main")
def index():
    return render_template('system/camera/camera_add.html')

#   上传接口


@bp.post('/add')
@authorize("system:cameraadd:add", log=True)
def upload_api():
    print("调用成功..................")
    # if 'file' in request.files:
    #     photo = request.files['file']
    #     mime = request.files['file'].content_type

    #     file_url = upload_curd.upload_one(photo=photo, mime=mime)
    #     res = {
    #         "msg": "上传成功",
    #         "code": 0,
    #         "success": True,
    #         "data":
    #             {"src": file_url}
    #     }
    #     return jsonify(res)
    return fail_api()
