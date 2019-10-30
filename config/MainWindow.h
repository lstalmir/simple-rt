#pragma once
#include "ConsoleWindow.h"
#include "FramebufferWindow.h"

#include <QMainWindow>
#include <QProcess>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void runApplication();
    void applicationFinished( int exitCode, QProcess::ExitStatus status );

private:
    Ui::MainWindow *ui;

    ConsoleWindow consoleWindow;
    FramebufferWindow framebufferWindow;

    QProcess applicationProc;

    QStringList collectArguments() const;
};

