# rust_disk_backend

### Build setup
1. Install the [rust compiler toolchain](https://rust-lang.org/tools/install/).
2. Install [maturin](https://www.maturin.rs/) into your venv.
3. After activating your venv, and from the `rust_disk_backend` directory run `maturin develop --release`.

### Configuration & Running
* Pass `LMCACHE_ENABLE_LOCAL_DISK=0` to use the rust disk backend, pass `LMCACHE_ENABLE_LOCAL_DISK=1` to use the default python backend.
* Use `RUST_LOG` to enable rust logging, e.g. `RUST_LOG=debug`.
* Set `num_shards` in the config file to configure number of shards for the engine.
* Set `max_container_size` (in megabytes) in the  file to set the size of each cache file.
