// AllToAll

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void AllToAll(T const* sbuf, int /*sc*/, T* rbuf, int rc, Comm comm,
              SyncInfo<D> const&)
{
    EL_DEBUG_CSE
    if (rc == 0)
        return;

    using Backend = BestBackend<T,D,Collective::ALLTOALL>;
    // FIXME Synchronize
    Al::Alltoall<Backend>(
        sbuf, rbuf, rc, comm.template GetComm<Backend>());
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void AllToAll(T const* sbuf, int /*sc*/, T* rbuf, int rc, Comm comm,
              SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    if (rc == 0)
        return;

    using Backend = BestBackend<T,Device::GPU,Collective::ALLTOALL>;
    SyncInfo<Device::GPU> alSyncInfo(comm.template GetComm<Backend>().get_stream(),
                                     syncInfo.event_);

    auto syncHelper = MakeMultiSync(alSyncInfo, syncInfo);

    Al::Alltoall<Backend>(
        sbuf, rbuf, rc, comm.template GetComm<Backend>());

}

#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                  Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllToAll(T const* sbuf, int sc, T* rbuf, int rc, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto size_c = Size(comm);

    ENSURE_HOST_SEND_BUFFER(sbuf, sc*size_c, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc*size_c, syncInfo);
#endif

    Synchronize(syncInfo);
    CheckMpi(
        MPI_Alltoall(
            sbuf, sc, TypeMap<T>(),
            rbuf, rc, TypeMap<T>(), comm.comm));
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=EnableIf<IsPacked<T>>*/>
void AllToAll(Complex<T> const* sbuf,
              int sc, Complex<T>* rbuf, int rc, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    auto size_c = Size(comm);

    ENSURE_HOST_SEND_BUFFER(sbuf, sc*size_c, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, rc*size_c, syncInfo);
#endif

    Synchronize(syncInfo);

#ifdef EL_AVOID_COMPLEX_MPI
    CheckMpi(
        MPI_Alltoall(
            const_cast<Complex<T>*>(sbuf),
            2*sc, TypeMap<T>(),
            rbuf,
            2*rc, TypeMap<T>(), comm.comm));
#else
    CheckMpi(
        MPI_Alltoall(
            const_cast<Complex<T>*>(sbuf),
            sc, TypeMap<Complex<T>>(),
            rbuf,
            rc, TypeMap<Complex<T>>(), comm.comm));
#endif
}

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=DisableIf<IsPacked<T>>*/,
          typename/*=void*/>
void AllToAll(T const* sbuf, int sc, T* rbuf, int rc, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    const int commSize = mpi::Size(comm);
    const int totalSend = sc*commSize;
    const int totalRecv = rc*commSize;

#ifdef HYDROGEN_ENSURE_HOST_MPI_BUFFERS
    ENSURE_HOST_SEND_BUFFER(sbuf, totalSend, syncInfo);
    ENSURE_HOST_RECV_BUFFER(rbuf, totalRecv, syncInfo);
#endif

    Synchronize(syncInfo);

    std::vector<byte> packedSend, packedRecv;
    Serialize(totalSend, sbuf, packedSend);
    ReserveSerialized(totalRecv, rbuf, packedRecv);
    CheckMpi(
        MPI_Alltoall(
            packedSend.data(), sc, TypeMap<T>(),
            packedRecv.data(), rc, TypeMap<T>(), comm.comm));
    Deserialize(totalRecv, packedRecv, rbuf);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                              Not<IsAluminumSupported<T,D,COLL>>>>*/,
          typename/*=void*/, typename/*=void*/, typename/*=void*/>
void AllToAll(T const*, int, T*, int, Comm, SyncInfo<D> const&)
{
    LogicError("AllToAll: Bad device/type combination.");
}

#define MPI_ALLTOALL_PROTO_DEV(T,D) \
    template void AllToAll(T const*, int, T*, int, Comm, SyncInfo<D> const&);

#ifndef HYDROGEN_HAVE_CUDA
#define MPI_ALLTOALL_PROTO(T)             \
    MPI_ALLTOALL_PROTO_DEV(T,Device::CPU)
#else
#define MPI_ALLTOALL_PROTO(T)             \
    MPI_ALLTOALL_PROTO_DEV(T,Device::CPU) \
    MPI_ALLTOALL_PROTO_DEV(T,Device::GPU)
#endif // HYDROGEN_HAVE_CUDA

MPI_ALLTOALL_PROTO(byte)
MPI_ALLTOALL_PROTO(int)
MPI_ALLTOALL_PROTO(unsigned)
MPI_ALLTOALL_PROTO(long int)
MPI_ALLTOALL_PROTO(unsigned long)
MPI_ALLTOALL_PROTO(float)
MPI_ALLTOALL_PROTO(double)
MPI_ALLTOALL_PROTO(long long int)
MPI_ALLTOALL_PROTO(unsigned long long)
MPI_ALLTOALL_PROTO(ValueInt<Int>)
MPI_ALLTOALL_PROTO(Entry<Int>)
MPI_ALLTOALL_PROTO(Complex<float>)
MPI_ALLTOALL_PROTO(ValueInt<float>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<float>>)
MPI_ALLTOALL_PROTO(Entry<float>)
MPI_ALLTOALL_PROTO(Entry<Complex<float>>)
MPI_ALLTOALL_PROTO(Complex<double>)
MPI_ALLTOALL_PROTO(ValueInt<double>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<double>>)
MPI_ALLTOALL_PROTO(Entry<double>)
MPI_ALLTOALL_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_ALLTOALL_PROTO(DoubleDouble)
MPI_ALLTOALL_PROTO(QuadDouble)
MPI_ALLTOALL_PROTO(Complex<DoubleDouble>)
MPI_ALLTOALL_PROTO(Complex<QuadDouble>)
MPI_ALLTOALL_PROTO(ValueInt<DoubleDouble>)
MPI_ALLTOALL_PROTO(ValueInt<QuadDouble>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_ALLTOALL_PROTO(Entry<DoubleDouble>)
MPI_ALLTOALL_PROTO(Entry<QuadDouble>)
MPI_ALLTOALL_PROTO(Entry<Complex<DoubleDouble>>)
MPI_ALLTOALL_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_ALLTOALL_PROTO(Quad)
MPI_ALLTOALL_PROTO(Complex<Quad>)
MPI_ALLTOALL_PROTO(ValueInt<Quad>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<Quad>>)
MPI_ALLTOALL_PROTO(Entry<Quad>)
MPI_ALLTOALL_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_ALLTOALL_PROTO(BigInt)
MPI_ALLTOALL_PROTO(BigFloat)
MPI_ALLTOALL_PROTO(Complex<BigFloat>)
MPI_ALLTOALL_PROTO(ValueInt<BigInt>)
MPI_ALLTOALL_PROTO(ValueInt<BigFloat>)
MPI_ALLTOALL_PROTO(ValueInt<Complex<BigFloat>>)
MPI_ALLTOALL_PROTO(Entry<BigInt>)
MPI_ALLTOALL_PROTO(Entry<BigFloat>)
MPI_ALLTOALL_PROTO(Entry<Complex<BigFloat>>)
#endif

} // namespace mpi
} // namespace El