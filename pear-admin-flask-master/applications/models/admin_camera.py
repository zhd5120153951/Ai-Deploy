import datetime
from applications.extensions import db


class Camera(db.Model):
    __tablename__ = 'admin_camera'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.Integer, comment="摄像头状态")
    create_time = db.Column(db.DateTime, default=datetime.datetime.now)
