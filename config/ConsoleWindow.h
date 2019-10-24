#pragma once

#include <QMainWindow>
#include <QMutex>

namespace Ui {
    class ConsoleWindow;
}

class ConsoleWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit ConsoleWindow(QWidget *parent = nullptr);
    ~ConsoleWindow();

    void clear();
    void print( QString message );

private:
    Ui::ConsoleWindow *ui;

    QMutex consoleMutex;
};

