#pragma once

#include <QMainWindow>

namespace Ui {
    class FramebufferWindow;
}

class FramebufferWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit FramebufferWindow( QWidget* parent = nullptr );
    ~FramebufferWindow();

    void openImage( QString path );

private:
    Ui::FramebufferWindow *ui;
};

