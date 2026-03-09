import os
import socket
from typing import Any, Dict, Optional


def remove_glob_syntax(path: str) -> str:
    """Remove glob syntax from a file path."""
    return path.replace("*", "").replace("?", "")


class Singleton(type):
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def instance(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return cls(*args, **kwargs)

    def has(cls: Any) -> bool:
        return cls in cls._instances

    def reset(cls: Any) -> None:
        cls._instances = {}

    def remove(cls: Any) -> None:
        if cls in cls._instances:
            del cls._instances[cls]


def is_package_avail(name: str) -> bool:
    import importlib.util
    import sys

    if name in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if spec.loader:
            spec.loader.exec_module(module)
            return True
    return False


class MPIUtils(metaclass=Singleton):
    def __init__(self):
        if Singleton.has(MPIUtils):
            raise Exception("MPIUtils is a singleton class and cannot be instantiated more than once.")

        self._rank: int = 0
        self._local_rank: int = 0
        self._size: int = 1
        self._num_nodes: int = 1
        self._ppn: int = 1
        self._is_initialized: bool = False
        self._is_mpi = False
        self._comm_world = None
        self._comm_local = None

    @staticmethod
    def instance(*args: Any, **kwargs: Any) -> "MPIUtils":
        instance = Singleton.instance(MPIUtils, *args, **kwargs)
        return instance

    def _initialize(self):
        if self._is_initialized:
            return

        if is_package_avail("mpi4py"):
            from mpi4py import MPI

            if not MPI.Is_initialized():
                MPI.Init()

            self._comm_world = MPI.COMM_WORLD
            self._comm_local = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
            self._rank = self._comm_world.Get_rank()
            self._size = self._comm_world.Get_size()
            self._local_rank = self._comm_local.Get_rank()
            self._ppn = self._comm_local.Get_size()
            self._num_nodes = self._size // self._ppn
            self._is_initialized = True
            self._is_mpi = True
        else:
            self._rank = int(os.environ.get("FLUX_TASK_RANK", 0))
            self._size = int(os.environ.get("FLUX_JOB_SIZE", 1))
            self._local_rank = int(os.environ.get("FLUX_TASK_LOCAL_ID", 0))
            self._num_nodes = int(os.environ.get("FLUX_JOB_NNODES", 1))
            self._is_initialized = True
            self._is_mpi = False

    @staticmethod
    def is_initialized() -> bool:
        return MPIUtils.instance()._is_initialized

    @staticmethod
    def is_mpi_initialized() -> bool:
        return MPIUtils.instance()._is_mpi and MPIUtils.instance()._is_initialized

    @staticmethod
    def initialize() -> "MPIUtils":
        instance = MPIUtils.instance()
        instance._initialize()
        return instance

    @staticmethod
    def rank() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing rank."
        return MPIUtils.instance()._rank

    @staticmethod
    def local_rank() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing local rank."
        return MPIUtils.instance()._local_rank

    @staticmethod
    def size() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing world size."
        return MPIUtils.instance()._size

    @staticmethod
    def world_size() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing world size."
        return MPIUtils.instance()._size

    @staticmethod
    def num_nodes() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing number of nodes."
        return MPIUtils.instance()._num_nodes

    @staticmethod
    def ppn() -> int:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing processes per node."
        return MPIUtils.instance()._ppn

    @staticmethod
    def is_mpi() -> bool:
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before checking if MPI is used."
        return MPIUtils.instance()._is_mpi

    @staticmethod
    def comm_world():
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing the world communicator."
        assert MPIUtils.is_mpi(), "MPIUtils must be initialized with MPI to access the world communicator."
        return MPIUtils.instance()._comm_world

    @staticmethod
    def comm_local():
        assert MPIUtils.is_initialized(), "MPIUtils must be initialized before accessing the local communicator."
        assert MPIUtils.is_mpi(), "MPIUtils must be initialized with MPI to access the local communicator."
        return MPIUtils.instance()._comm_local

    @staticmethod
    def finalize():
        instance = MPIUtils.instance()
        if instance._is_initialized and instance._is_mpi:
            from mpi4py import MPI

            if MPI.Is_initialized():
                MPI.Finalize()
        instance._is_initialized = False
        instance._is_mpi = False
        instance._comm_world = None
        instance._comm_local = None
        Singleton.remove(MPIUtils)


def get_rank():
    if not MPIUtils.is_initialized():
        MPIUtils.initialize()
    return MPIUtils.rank()


def get_local_rank():
    if not MPIUtils.is_initialized():
        MPIUtils.initialize()
    return MPIUtils.local_rank()


def get_world_size():
    if not MPIUtils.is_initialized():
        MPIUtils.initialize()
    return MPIUtils.size()


def find_free_network_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_free_network_port_if_taken(port: Optional[int] = None) -> int:
    if port is None:
        return find_free_network_port()

    if is_port_in_use(port):
        return find_free_network_port()

    return port


def get_hostname() -> str:
    return socket.gethostname()


def get_master_addr_and_port(port: Optional[int] = None, lc_machine: bool = True, set_env: bool = True) -> tuple[str, int]:
    if MPIUtils.is_mpi_initialized():
        if MPIUtils.rank() == 0:
            hostname = get_hostname()
            if lc_machine:
                hostname = lc_machine_full_hostname(hostname)
            port = find_free_network_port_if_taken(port)
        else:
            hostname = None
            port = None
        hostname = MPIUtils.comm_world().bcast(hostname, root=0)
        port = MPIUtils.comm_world().bcast(port, root=0)
        if set_env:
            os.environ["MASTER_ADDR"] = hostname
            os.environ["MASTER_PORT"] = str(port)
    else:
        hostname = get_hostname()
        if lc_machine:
            hostname = lc_machine_full_hostname(hostname)
        port = find_free_network_port_if_taken(port)
        if set_env:
            os.environ["MASTER_ADDR"] = hostname
            os.environ["MASTER_PORT"] = str(port)
    return hostname, port


def lc_machine_full_hostname(hostname: str):
    return f"{hostname}.llnl.gov"
