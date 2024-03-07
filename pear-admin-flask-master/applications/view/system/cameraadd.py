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
