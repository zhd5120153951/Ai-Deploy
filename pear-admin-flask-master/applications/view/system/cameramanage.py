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


#  图片数据--非此功能可以忽略
@bp.get('/table')
@authorize("system:cameramanage:add")
def table():
    page = request.args.get('page', type=int)
    limit = request.args.get('limit', type=int)
    data, count = upload_curd.get_photo(page=page, limit=limit)
    return table_api(data=data, count=count)


#   新增界面
@bp.get('/add')
@authorize("system:cameramanage:add", log=True)
def upload():
    return render_template('system/camera/camera_add.html')
