# the use of github

# 1. config
git config --global user.name "your name" 
git config --global user.email "your email"

# 2. new project
cd /path              //首先指定到你的项目目录下
git init
touch README.md
git add README.md
git commit -m "first commit"
//用你仓库的url,vscode.git中vscode为仓库名称,使用时必须先创建
git remote add origin https://github.com/yourname/pro.git   
git push -u origin master  //提交到你的仓库

# 3. push
git commit -am "some str" 
git push