from mpi4py import MPI
import numpy as np
import time

def jacobi_serial(nx, ny, max_iter, tolerance):
    u = np.zeros((ny, nx), dtype=np.float64)
    u_new = np.zeros((ny, nx), dtype=np.float64)
    
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    
    u_new[:, :] = u[:, :]
    
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    alpha = dx**2 * dy**2 / (2 * (dx**2 + dy**2))
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    X, Y = np.meshgrid(x,y,indexing='xy')
    f  = np.sin(np.pi*X) * np.sin(np.pi*Y)

    for iteration in range(max_iter):
        max_diff = 0.0
        
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u_new[j, i] = alpha * (
                    (u[j, i + 1] + u[j, i - 1]) / (dx**2) +
                    (u[j + 1, i] + u[j - 1, i]) / (dy**2) - f[j,i]*alpha
                )
                
                diff = abs(u_new[j, i] - u[j, i])
                if diff > max_diff:
                    max_diff = diff
        
        u[1:-1, 1:-1] = u_new[1:-1, 1:-1]
        
        if max_diff < tolerance:
            print(f"erro global: {max_diff_global}")
            print(f"iteração: {iteration+1}")
            break
        if iteration == (max_iter-1):
            print(f"iteração: {iteration+1}. Atingiu o limite de iterações.")
    
    return u, iteration + 1

def jacobi_parallel(nx, ny, max_iter, tolerance):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Criar topologia cartesiana 2D
    dims = MPI.Compute_dims(size, 2)
    periods = [False, False]
    reorder = True
    
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)
    rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(rank)
    
    nprocs_x, nprocs_y = dims
    coord_x, coord_y = coords
    
    # Calcular dimensões locais
    nx_local = nx // nprocs_x
    ny_local = ny // nprocs_y
    
    remainder_x = nx % nprocs_x
    remainder_y = ny % nprocs_y
    
    if coord_x < remainder_x:
        nx_local += 1
        start_x = coord_x * nx_local
    else:
        start_x = coord_x * nx_local + remainder_x
    
    if coord_y < remainder_y:
        ny_local += 1
        start_y = coord_y * ny_local
    else:
        start_y = coord_y * ny_local + remainder_y
    
    end_x = start_x + nx_local
    end_y = start_y + ny_local
    
    # Encontrar vizinhos
    left_neighbor, right_neighbor = cart_comm.Shift(0, 1)#(direction, step) #Direção 0: Eixo x. 
    top_neighbor, bottom_neighbor = cart_comm.Shift(1, 1)                   #Direção 1: Eixo y.
    
    # Matriz local com células fantasmas
    u_local = np.zeros((ny_local + 2, nx_local + 2), dtype=np.float64)
    u_new_local = np.zeros((ny_local + 2, nx_local + 2), dtype=np.float64)
    
    # Aplicar condições de contorno globais
    if start_x == 0:
        u_local[1:-1, 1] = 0.0
    if end_x == nx:
        u_local[1:-1, -2] = 0.0
    if start_y == 0:
        u_local[1, 1:-1] = 0.0
    if end_y == ny:
        u_local[-2, 1:-1] = 0.0
    
    u_new_local[:, :] = u_local[:, :]
    
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    alpha = dx * dx * dy * dy / (2 * (dx * dx + dy * dy))
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    X, Y = np.meshgrid(x,y,indexing='xy')
    f  = np.sin(np.pi*X) * np.sin(np.pi*Y)

    for iteration in range(max_iter):
        requests = []
        
        # Enviar/receber bordas esquerda/direita
        if left_neighbor != MPI.PROC_NULL:
            left_send = u_local[1:-1, 1].copy()
            requests.append(cart_comm.Isend(left_send, dest=left_neighbor, tag=1))
        
        if right_neighbor != MPI.PROC_NULL:
            right_send = u_local[1:-1, -2].copy()
            requests.append(cart_comm.Isend(right_send, dest=right_neighbor, tag=2))
        
        if left_neighbor != MPI.PROC_NULL:
            left_recv = np.empty(ny_local, dtype=np.float64)
            requests.append(cart_comm.Irecv(left_recv, source=left_neighbor, tag=2))
        
        if right_neighbor != MPI.PROC_NULL:
            right_recv = np.empty(ny_local, dtype=np.float64)
            requests.append(cart_comm.Irecv(right_recv, source=right_neighbor, tag=1))
        
        # Enviar/receber bordas superior/inferior
        if top_neighbor != MPI.PROC_NULL:
            top_send = u_local[1, 1:-1].copy()
            requests.append(cart_comm.Isend(top_send, dest=top_neighbor, tag=3))
        
        if bottom_neighbor != MPI.PROC_NULL:
            bottom_send = u_local[-2, 1:-1].copy()
            requests.append(cart_comm.Isend(bottom_send, dest=bottom_neighbor, tag=4))
        
        if top_neighbor != MPI.PROC_NULL:
            top_recv = np.empty(nx_local, dtype=np.float64)
            requests.append(cart_comm.Irecv(top_recv, source=top_neighbor, tag=4))
        
        if bottom_neighbor != MPI.PROC_NULL:
            bottom_recv = np.empty(nx_local, dtype=np.float64)
            requests.append(cart_comm.Irecv(bottom_recv, source=bottom_neighbor, tag=3))
        
        # Aguardar conclusão de todas as comunicações
        MPI.Request.Waitall(requests)
        
        # Atualizar células fantasmas com dados recebidos
        if left_neighbor != MPI.PROC_NULL:
            u_local[1:-1, 0] = left_recv
        if right_neighbor != MPI.PROC_NULL:
            u_local[1:-1, -1] = right_recv
        if top_neighbor != MPI.PROC_NULL:
            u_local[0, 1:-1] = top_recv
        if bottom_neighbor != MPI.PROC_NULL:
            u_local[-1, 1:-1] = bottom_recv
        
        # Calcular novos valores
        max_diff_local = 0.0
        
        for j in range(1, ny_local + 1):
            for i in range(1, nx_local + 1):
                global_i = start_x + i - 1
                global_j = start_y + j - 1
                
                if (global_i > 0 and global_i < nx - 1 and 
                    global_j > 0 and global_j < ny - 1):
                    
                    u_new_local[j, i] = alpha * (
                        (u_local[j, i + 1] + u_local[j, i - 1]) / (dx * dx) +
                        (u_local[j + 1, i] + u_local[j - 1, i]) / (dy * dy) -
                         f[j,i]
                    )
                    
                    diff = abs(u_new_local[j, i] - u_local[j, i])
                    if diff > max_diff_local:
                        max_diff_local = diff
        
        u_local[1:-1, 1:-1] = u_new_local[1:-1, 1:-1]
        
        max_diff_global = cart_comm.allreduce(max_diff_local, op=MPI.MAX)
        
        if max_diff_global < tolerance:
            print(f"erro global: {max_diff_global}")
            print(f"iteração: {iteration+1}")
            break
        if iteration == (max_iter-1):
            print(f"iteração: {iteration+1}. Atingiu o limite de iterações.")
 
    # Coletar resultados
    local_data = u_local[1:-1, 1:-1].copy()
    
    if rank == 0:
        global_u = np.zeros((ny, nx), dtype=np.float64)
        counts = []
        displacements = []
        current_displacement = 0
        
        for r in range(size):
            coords_r = cart_comm.Get_coords(r)
            coord_x, coord_y = coords_r
            
            nx_loc = nx // nprocs_x
            ny_loc = ny // nprocs_y
            
            rem_x = nx % nprocs_x
            rem_y = ny % nprocs_y
            
            if coord_x < rem_x:
                nx_loc += 1
                start_x = coord_x * nx_loc
            else:
                start_x = coord_x * nx_loc + rem_x
            
            if coord_y < rem_y:
                ny_loc += 1
                start_y = coord_y * ny_loc
            else:
                start_y = coord_y * ny_loc + rem_y
            
            count = nx_loc * ny_loc
            counts.append(count)
            displacements.append(current_displacement)
            current_displacement += count
        
        recv_buffer = np.empty(np.sum(counts), dtype=np.float64)
    else:
        global_u = None
        counts = None
        displacements = None
        recv_buffer = None
    
    cart_comm.Gatherv(local_data.flatten(), [recv_buffer, counts, displacements, MPI.DOUBLE], root=0)
    
    if rank == 0:
        for r in range(size):
            coords_r = cart_comm.Get_coords(r)
            coord_x, coord_y = coords_r
            
            nx_loc = nx // nprocs_x
            ny_loc = ny // nprocs_y
            
            rem_x = nx % nprocs_x
            rem_y = ny % nprocs_y
            
            if coord_x < rem_x:
                nx_loc += 1
                start_x = coord_x * nx_loc
            else:
                start_x = coord_x * nx_loc + rem_x
            
            if coord_y < rem_y:
                ny_loc += 1
                start_y = coord_y * ny_loc
            else:
                start_y = coord_y * ny_loc + rem_y
            
            start_idx = displacements[r]
            end_idx = start_idx + nx_loc * ny_loc
            proc_data = recv_buffer[start_idx:end_idx].reshape(ny_loc, nx_loc)
            
            global_u[start_y:start_y+ny_loc, start_x:start_x+nx_loc] = proc_data
    
    return (global_u, iteration + 1) if rank == 0 else (None, iteration + 1)

def run_scaling_analysis():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    num_runs = 5
    max_iter = 1000      #1000
    tolerance = 1e-4   #1e-6
    
    if rank == 0:
        print(f"=== ANÁLISE MÉTODO DE JACOBI ===")
        print(f"Processos: {size}")
        print(f"Iterações máximas: {max_iter}")
        print(f"Tolerância: {tolerance}")
        print(f"Execuções por teste: {num_runs}")
        print()
    
    # Strong Scaling
    if rank == 0:
        print("=== STRONG SCALING ===")
    
    strong_nx, strong_ny = 200, 200 #2000 , 2000
    
    if rank == 0:
        print(f"Tamanho do problema: {strong_nx}x{strong_ny}")
        
        serial_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            result, iterations = jacobi_serial(strong_nx, strong_ny, max_iter, tolerance)
            end_time = time.perf_counter()
            serial_times.append(end_time - start_time)
        
        serial_time = np.mean(serial_times[1:])
        print(f"Serial: {serial_time:.4f}s ({iterations} iterações)")
    
    comm.Barrier()
    
    parallel_times = []
    for i in range(num_runs):
        comm.Barrier()
        start_time = time.perf_counter()
        result, iterations = jacobi_parallel(strong_nx, strong_ny, max_iter, tolerance)
        end_time = time.perf_counter()
        
        if rank == 0:
            parallel_times.append(end_time - start_time)
    
    if rank == 0:
        parallel_time = np.mean(parallel_times[1:]) if parallel_times else serial_time
        speedup = serial_time / parallel_time
        efficiency = (speedup / size) * 100
        
        print(f"Paralelo: {parallel_time:.4f}s ({iterations} iterações)")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Eficiência: {efficiency:.1f}%")
        print()
    
    # Weak Scaling
    if rank == 0:
        print("=== WEAK SCALING ===")
    
    base_nx, base_ny = 200,200#500, 500
    weak_nx, weak_ny = base_nx * int(np.sqrt(size)), base_ny * int(np.sqrt(size))
    
    if rank == 0:
        print(f"Tamanho por processo: {base_nx}x{base_ny}")
        print(f"Tamanho total: {weak_nx}x{weak_ny}")
    
    comm.Barrier()
    
    weak_times = []
    for i in range(num_runs):
        comm.Barrier()
        start_time = time.perf_counter()
        result, iterations = jacobi_parallel(weak_nx, weak_ny, max_iter, tolerance)
        end_time = time.perf_counter()
        
        if rank == 0:
            weak_times.append(end_time - start_time)
    
    if rank == 0:
        weak_time = np.mean(weak_times[1:]) if weak_times else 0
        print(f"Tempo weak scaling: {weak_time:.4f}s ({iterations} iterações)")
        print()

if __name__ == "__main__":
    run_scaling_analysis()
