#include "FramebufferWindow.h"
#include "ui_FramebufferWindow.h"
#include <QPixmap>

FramebufferWindow::FramebufferWindow( QWidget* parent )
    : QMainWindow( parent )
    , ui( new Ui::FramebufferWindow )
{
    ui->setupUi( this );
}

FramebufferWindow::~FramebufferWindow()
{
    delete ui;
}

void FramebufferWindow::openImage( QString path )
{
    ui->image->setPixmap( path );

    // Resize the window
    resize( ui->image->pixmap()->width(),
        ui->image->pixmap()->height() );

    // Set window title
    setWindowTitle( path + " - Framebuffer" );
}
