from flask import Flask, request, jsonify, flash, session, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash

import sqlite3


app = Flask(__name__)
CORS(app)
api = Api(app)
# 配置密钥,用于加密session数据
# app.config['SECREAT_KEY'] = "daito_yolov5_flask"
app.secret_key = "daito_yolov5_flask"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
jwt = JWTManager(app)
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(50))


class ProcessDataSource(Resource):
    @jwt_required()
    def post(self):
        print("接收到前端请求了...")
        # 处理登录逻辑
        username = request.json['username']
        print(username)
        password = request.json['password']
        print(password)

        # 假设验证通过
        # 在实际情况中，你需要根据你的具体逻辑进行身份验证
        # 如果验证失败，可以返回错误信息给前端
        # user = User.query.filter_by(username=username).first()
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE username = ?', (username, ))
            isExistUser = cursor.fetchone()
            conn.commit()
        if isExistUser and check_password_hash(isExistUser[2], password):
            session['username'] = username
            # 登录成功，返回认证通过的消息给前端
            response = {'message': 'Login Successful'}
            conn.close()
            # return jsonify(response), 200#这里给前端就不给消息了
        return redirect(url_for('homepage'))


class SuccessResource(Resource):
    @jwt_required()
    def get(self):
        return render_template('homepage.html')


@app.route('/login', methods=['POST'])
def login():
    print("接收到前端请求了...")
    # 处理登录逻辑
    username = request.json['username']
    print(username)
    password = request.json['password']
    print(password)

    # 假设验证通过
    # 在实际情况中，你需要根据你的具体逻辑进行身份验证
    # 如果验证失败，可以返回错误信息给前端
    # user = User.query.filter_by(username=username).first()
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE username = ?', (username, ))
        isExistUser = cursor.fetchone()

        conn.commit()
    if isExistUser and check_password_hash(isExistUser[2], password):
        session['username'] = username
        # 登录成功，返回认证通过的消息给前端
        response = {'message': 'Login Successful'}
        conn.close()
        # return jsonify(response), 200#这里给前端就不给消息了
        return redirect(url_for('homepage'))
    else:
        flash('Invalid username or password', 'info')
        conn.close()
        return jsonify({'message': 'Invalid password'}), 401


@app.route('/homepage')
def a():
    return render_template('homepage.html')


if __name__ == "__main__":
    api.add_resource(ProcessDataSource, '/login')
    api.add_resource(SuccessResource, '/homepage')
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
