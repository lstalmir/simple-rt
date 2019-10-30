#include "ConsoleWindow.h"
#include "ui_ConsoleWindow.h"
#include <QMutexLocker>

ConsoleWindow::ConsoleWindow( QWidget* parent )
    : QMainWindow( parent )
    , ui( new Ui::ConsoleWindow )
{
    ui->setupUi( this );
    clear();
}

ConsoleWindow::~ConsoleWindow()
{
    delete ui;
}

void ConsoleWindow::clear()
{
    QMutexLocker lk( &consoleMutex );
    ui->console->clear();
}

void ConsoleWindow::print( QString message )
{
    QMutexLocker lk( &consoleMutex );
    ui->console->appendPlainText( message );
}
