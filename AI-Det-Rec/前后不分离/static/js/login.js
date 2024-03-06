// 暂时不用
document.getElementById('loginForm').addEventListener('submit', function (event) {
    event.preventDefault(); // 阻止表单的默认提交行为

    var username = document.getElementById('username').value.trim();
    var password = document.getElementById('password').value.trim();

    // 判断用户名和密码是否合规
    if (username.length < 5 || username.length > 20) {
        document.getElementById('message').textContent = '用户名长度应为5到20个字符';
    } else if (password.length < 8 || password.length > 20) {
        document.getElementById('message').textContent = '密码长度应为8到20个字符';
    } else {
        // 登录成功
        // alert('登录成功！');
        // document.getElementById("message").textContent = "登陆成功";
        // 登录成功，跳转到主界面
        window.location.href = 'homepage';
    }
});
