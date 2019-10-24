#pragma once

#include <QWidget>

namespace Ui {
    class FileEdit;
}

class FileEdit : public QWidget
{
    Q_OBJECT

public:
    explicit FileEdit(QWidget *parent = nullptr);
    ~FileEdit();

private:
    Ui::FileEdit *ui;
};

