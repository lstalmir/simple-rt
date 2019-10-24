#include "FileEdit.h"
#include "ui_FileEdit.h"

FileEdit::FileEdit(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FileEdit)
{
    ui->setupUi(this);
}

FileEdit::~FileEdit()
{
    delete ui;
}
