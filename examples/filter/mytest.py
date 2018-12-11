import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.NgSpice.Shared import NgSpiceShared

ngspice = NgSpiceShared.new_instance()

print(ngspice.exec_command('version -f'))
print(ngspice.exec_command('print all'))
print(ngspice.exec_command('devhelp resistor'))