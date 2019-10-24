#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    consoleWindow(this)
{
    ui->setupUi(this);

    ui->mode_comboBox->clear();
    ui->mode_comboBox->addItem( "", "" );
    ui->mode_comboBox->addItem( "OpenMP", "-openmp" );
    ui->mode_comboBox->addItem( "OpenCL", "-opencl" );
    ui->mode_comboBox->addItem( "Test", "-t" );

    connect( &applicationProc, SIGNAL( finished(int,QProcess::ExitStatus) ), this, SLOT( applicationFinished(int,QProcess::ExitStatus) ) );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::runApplication()
{
    if( applicationProc.state() == QProcess::ProcessState::Running )
    {
        // Process is already running
        return;

        // TODO: Message
    }

    // Update UI to avoid multiple instances of the raytracer in the same time
    ui->run_button->setEnabled( false );

    const QStringList arguments = collectArguments();
    const QString command = "simple-rt " + arguments.join( " " );

    consoleWindow.show();
    consoleWindow.clear();
    consoleWindow.print( "$ " + command + "\n" );

    applicationProc.start( "simple-rt", arguments );
}

void MainWindow::applicationFinished( int exitCode, QProcess::ExitStatus )
{
    ui->run_button->setEnabled( true );

    consoleWindow.print( applicationProc.readAll() );
    consoleWindow.print( "Application exited with code " + QString::number( exitCode ) + "\n" );
}

QStringList MainWindow::collectArguments() const
{
    QStringList arguments;

    if( !ui->input_lineEdit->text().isEmpty() )
    {
        arguments << "-i" << ui->input_lineEdit->text();
    }

    if( !ui->output_lineEdit->text().isEmpty() )
    {
        arguments << "-o" << ui->output_lineEdit->text();
    }

    if( !ui->mode_comboBox->currentData().toString().isEmpty() )
    {
        arguments << ui->mode_comboBox->currentData().toString();
    }

    if( !ui->width_lineEdit->text().isEmpty() )
    {
        arguments << "-w" << ui->width_lineEdit->text();
    }

    if( !ui->height_lineEdit->text().isEmpty() )
    {
        arguments << "-h" << ui->height_lineEdit->text();
    }

    if( !ui->boundingBoxes_checkBox->isChecked() )
    {
        arguments << "-disableboundingboxes";
    }

    if( ui->adjustAspect_checkBox->isChecked() )
    {
        arguments << "-adjustaspect";
    }

    if( !ui->simdInstructions_checkBox->isChecked() )
    {
        arguments << "-disableintrinsics";
    }

    return arguments;
}
