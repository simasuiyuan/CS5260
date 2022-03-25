
REPO_NAME=CS5260
REPO_URL=https://github.com/simasuiyuan/$REPO_NAME.git
WORK_DIR=CS5260
echo ">> Repository Name:"
echo $REPO_URL
if [[ -d $WORK_DIR ]]; then
  cd $WORK_DIR
  git pull "$REPO_URL"
else 
  git clone "$REPO_URL"
  cd $WORK_DIR
fi
echo "available branch:"
echo | git branch -a
echo "select branch:"
read BRANCH
git checkout $BRANCH
