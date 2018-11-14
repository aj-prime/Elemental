// Collective

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    using Backend = BestBackend<T,D,Collective::SENDRECV>;
    Al::SendRecv<Backend>(
        sbuf, sc, to, rbuf, rc, from, comm.template GetComm<Backend>());
}

template <typename T, Device D,
          typename/*=EnableIf<IsAluminumSupported<T,D,COLL>>*/>
void SendRecv(T* buf, int count, int to, int from, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    EL_DEBUG_CSE

    using Backend = BestBackend<T,D,Collective::SENDRECV>;
    // Not sure if Al is ok with this bit
    //std::cout << "WARNING: IN-PLACE SENDRECV." << std::endl;
    Al::SendRecv<Backend>(
        buf, count, to, buf, count, from, comm.template GetComm<Backend>());
}

#ifdef HYDROGEN_HAVE_CUDA
template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    using Backend = BestBackend<T,Device::GPU,Collective::SENDRECV>;

    SyncInfo<Device::GPU> alSyncInfo(
        comm.template GetComm<Backend>().get_stream(),
        syncInfo.event_);

    auto multisync = MakeMultiSync(alSyncInfo, syncInfo);
    Al::SendRecv<Backend>(
        sbuf, sc, to, rbuf, rc, from,
        comm.template GetComm<Backend>());
}

template <typename T,
          typename/*=EnableIf<IsAluminumSupported<T,Device::GPU,COLL>>*/>
void SendRecv(T* buf, int count, int to, int from, Comm comm,
              SyncInfo<Device::GPU> const& syncInfo)
{
    EL_DEBUG_CSE
    using Backend = BestBackend<T,Device::GPU,Collective::SENDRECV>;

    SyncInfo<Device::GPU> alSyncInfo(
        comm.template GetComm<Backend>().get_stream(),
        syncInfo.event_);

    auto multisync = MakeMultiSync(alSyncInfo, syncInfo);

    // Not sure if Al is ok with this bit
    Al::SendRecv<Backend>(
        buf, count, to, buf, count, from,
        comm.template GetComm<Backend>());
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    TaggedSendRecv(sbuf, sc, to, 0, rbuf, rc, from, ANY_TAG,
                   std::move(comm), syncInfo);
}

template <typename T, Device D,
          typename/*=EnableIf<And<Not<IsDeviceValidType<T,D>>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo)
{
    LogicError("SendRecv: Bad type/device combination.");
}


template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/>
void SendRecv( T* buf, int count, int to, int from, Comm comm,
               SyncInfo<D> const& syncInfo)
{ TaggedSendRecv(buf, count, to, 0, from, ANY_TAG, comm, syncInfo); }

template <typename T, Device D,
          typename/*=EnableIf<And<IsDeviceValidType<T,D>,
                                Not<IsAluminumSupported<T,D,COLL>>>*/,
          typename/*=void*/, typename/*=void*/>
void SendRecv( T* buf, int count, int to, int from, Comm comm,
               SyncInfo<D> const& syncInfo)
{
    LogicError("SendRecv: Bad type/device combination.");
}

template <typename T, Device D>
T SendRecv( T sb, int to, int from, Comm comm, SyncInfo<D> const& syncInfo )
{ return TaggedSendRecv( sb, to, 0, from, ANY_TAG, comm, syncInfo ); }

#define MPI_COLLECTIVE_PROTO_DEV(T,D) \
    template void SendRecv(                                             \
        const T* sbuf, int sc, int to,                                  \
        T* rbuf, int rc, int from, Comm comm, SyncInfo<D> const&);      \
    template T SendRecv(                                                \
        T sb, int to, int from, Comm comm, SyncInfo<D> const&);         \
    template void SendRecv(                                             \
        T* buf, int count, int to, int from, Comm comm,                 \
        SyncInfo<D> const&);

#ifndef HYDROGEN_HAVE_CUDA
#define MPI_COLLECTIVE_PROTO(T)             \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU)
#else
#define MPI_COLLECTIVE_PROTO(T)             \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::CPU) \
    MPI_COLLECTIVE_PROTO_DEV(T,Device::GPU)
#endif

MPI_COLLECTIVE_PROTO(byte)
MPI_COLLECTIVE_PROTO(int)
MPI_COLLECTIVE_PROTO(unsigned)
MPI_COLLECTIVE_PROTO(long int)
MPI_COLLECTIVE_PROTO(unsigned long)
MPI_COLLECTIVE_PROTO(float)
MPI_COLLECTIVE_PROTO(double)
MPI_COLLECTIVE_PROTO(long long int)
MPI_COLLECTIVE_PROTO(unsigned long long)
MPI_COLLECTIVE_PROTO(ValueInt<Int>)
MPI_COLLECTIVE_PROTO(Entry<Int>)
MPI_COLLECTIVE_PROTO(Complex<float>)
MPI_COLLECTIVE_PROTO(ValueInt<float>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<float>>)
MPI_COLLECTIVE_PROTO(Entry<float>)
MPI_COLLECTIVE_PROTO(Entry<Complex<float>>)
MPI_COLLECTIVE_PROTO(Complex<double>)
MPI_COLLECTIVE_PROTO(ValueInt<double>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<double>>)
MPI_COLLECTIVE_PROTO(Entry<double>)
MPI_COLLECTIVE_PROTO(Entry<Complex<double>>)
#ifdef HYDROGEN_HAVE_QD
MPI_COLLECTIVE_PROTO(DoubleDouble)
MPI_COLLECTIVE_PROTO(QuadDouble)
MPI_COLLECTIVE_PROTO(Complex<DoubleDouble>)
MPI_COLLECTIVE_PROTO(Complex<QuadDouble>)
MPI_COLLECTIVE_PROTO(ValueInt<DoubleDouble>)
MPI_COLLECTIVE_PROTO(ValueInt<QuadDouble>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<DoubleDouble>>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<QuadDouble>>)
MPI_COLLECTIVE_PROTO(Entry<DoubleDouble>)
MPI_COLLECTIVE_PROTO(Entry<QuadDouble>)
MPI_COLLECTIVE_PROTO(Entry<Complex<DoubleDouble>>)
MPI_COLLECTIVE_PROTO(Entry<Complex<QuadDouble>>)
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
MPI_COLLECTIVE_PROTO(Quad)
MPI_COLLECTIVE_PROTO(Complex<Quad>)
MPI_COLLECTIVE_PROTO(ValueInt<Quad>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<Quad>>)
MPI_COLLECTIVE_PROTO(Entry<Quad>)
MPI_COLLECTIVE_PROTO(Entry<Complex<Quad>>)
#endif
#ifdef HYDROGEN_HAVE_MPC
MPI_COLLECTIVE_PROTO(BigInt)
MPI_COLLECTIVE_PROTO(BigFloat)
MPI_COLLECTIVE_PROTO(Complex<BigFloat>)
MPI_COLLECTIVE_PROTO(ValueInt<BigInt>)
MPI_COLLECTIVE_PROTO(ValueInt<BigFloat>)
MPI_COLLECTIVE_PROTO(ValueInt<Complex<BigFloat>>)
MPI_COLLECTIVE_PROTO(Entry<BigInt>)
MPI_COLLECTIVE_PROTO(Entry<BigFloat>)
MPI_COLLECTIVE_PROTO(Entry<Complex<BigFloat>>)
#endif

#undef MPI_COLLECTIVE_PROTO
#undef MPI_COLLECTIVE_PROTO_DEV
} // namespace mpi
} // namespace El