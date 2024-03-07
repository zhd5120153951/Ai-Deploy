import os
from flask import Blueprint, request, render_template, jsonify, current_app

from applications.common.utils.http import fail_api, success_api, table_api
from applications.common.utils.rights import authorize
from applications.extensions import db
from applications.models import Photo
from applications.common.utils import upload as upload_curd

bp = Blueprint('cameramanage', __name__, url_prefix='/cameramanage')


#  图片上传管理界面
@bp.get('/')
@authorize("system:cameramanage:main")
def index():
    return render_template('system/camera/camera_manage.html')

#  获取已添加的摄像头数据


@bp.get('/table')
@authorize("system:cameramanage:main")
def table():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)
    data, count = upload_curd.get_photo(page=page, limit=limit)
    return table_api(data=data, count=count)

#  删除逻辑


@bp.route('/delete', methods=['GET', 'POST'])
@authorize("system:cameramanage:delete", log=True)
def deletecamera():
    _id = request.form.get('id')
    res = upload_curd.delete_photo_by_id(_id)
    if res:
        return success_api(msg="删除成功")
    else:
        return fail_api(msg="删除失败")


@bp.route('/delete_batch', methods=['GET', 'POST'])
@authorize("system:cameramanage:delete", log=True)
def deletecamera_batch():
    pass
