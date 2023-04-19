# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

using Distributed
addprocs(1)
@everywhere using Yao
@everywhere using Yao.EasyBuild
# using Flux: Optimise
@everywhere using PyCall
@everywhere using Yao.YaoBlocks.AD
# @everywhere using SciPy
@everywhere using TimerOutputs
@everywhere minimize=pyimport("scipy.optimize").minimize
@pyimport numpy as np
# this should add to the sys python path.
@pyimport libdmet_qc.utils.operator_pool_helper as op 


@everywhere function get_fermion_hamiltonian(n::Int,terms::Array,coefs::Vector)
    gates=Dict("Z"=>Z,"X"=>X,"Y"=>Y)
    to_pauli(t::Tuple{Int,String})=put(n,t[1]+1=>get(gates,t[2],ErrorException("Invalid")))
    if(terms[1]==())
#         println("term1 ", terms[1], coefs[1])
        part0 = coefs[1]*put(n,1=>I2)
        hamiltonian = sum(coefs[2:end].*map(x->reduce(*, map(to_pauli,x)),terms[2:end]))
        return part0 + hamiltonian
    else
        hamiltonian = sum(coefs[1:end].*map(x->reduce(*, map(to_pauli,x)),terms[1:end]))
        return hamiltonian
    end

end

@everywhere gate_dict  = Dict("Rx"=> Yao.Rx,"Ry" => Yao.Ry,"Rz"=>Yao.Rz,"H"=>Yao.H, "CNOT"=>Yao.control)
@everywhere function gatelist_parse(n_qubits::Int,n_electrons::Union{Int, Vector{Int}},gate_list, initia_para_list, para_key_list)
    """Parse the gate list from python into Yao and using the NoParams module to label the gate that need 
    to evaluate gradients. The input are parameters with index from 0 to num-1.
    Then we construct the initial_para_dt for this one
    """
    initia_para_dt = Dict()
    for i=1:length(initia_para_list)
        # key = "p"*string(i-1)
        initia_para_dt[para_key_list[i]]=initia_para_list[i]
    end
    if typeof(n_electrons)==Int
        circuit = chain(n_qubits, repeat(X, 1:n_electrons)) # restricted circuit
    else
        n_alpha, n_beta = n_electrons[1], n_electrons[2]
        circuit = chain(n_qubits, repeat(X, 1:2:2*n_alpha-1))
        append!(circuit, chain(n_qubits, repeat(X, 2:2:2*n_beta)))
    end
    para_new_dt = Dict()
    cnt_new_para = 0
    for item in gate_list
        if (item[1]=="Rx" )|| (item[1]=="Ry" )
#             println(item[1])
            gate_name, para, idx = item
            append!(circuit, parse_rxry(n_qubits,gate_name, para, idx+1))
        elseif(item[1]=="H")
#             println("H gate")
            gate_name, idx = item
            append!(circuit,parse_single_qubit_gate(n_qubits,gate_name, idx+1))
        elseif(item[1]=="Rz")
#             println("Rz gate: ",Rz)
            #println(item)
            # println("Code being finish Rz.")
            cnt_new_para += 1
            gate_name, para_dt, idx = item # the para_dt store the corresponding parameters for this part
#             println("para_key_dt: ", para_dt)
            para_keys  = keys(para_dt)
            compose_para = 0
            para_new_dt[cnt_new_para] = para_dt
            # chain(num_qubits, time_evolve(hp, vars,tol=1e-5,check_hermicity=false))
            for single_key in para_keys
                # println("single keys: ", single_key)
                # actually this is the coefficient in front of the para
                compose_para += para_dt[single_key]* initia_para_dt[single_key]
            end
            #hp = compose_para*put(n_qubits, idx+1=>Z)
            # println("Code being finish Rz 2.")
            # println("compose_para: ",compose_para)
            #put the time to be 1 
            #append!(circuit,chain(n_qubits,time_evolve(hp, 1.0, tol=1e-5,check_hermicity=false)))
            append!(circuit,parse_rz(n_qubits, gate_name, 2*compose_para, idx+1))
            # parse the parameter gates
        elseif(item[1]=="CNOT")
#             println("CNOT gate")
            gate_name, ctr, target = item
            append!(circuit,parse_two_qubit_gate(n_qubits,gate_name, ctr+1, target+1))
        end 
    end
#     println("para_new_dt", length(para_new_dt))
    # println("circuit is: ", circuit)
    circuit, para_new_dt
end

# define several kinds of gate for parse Rx, Ry as one type, X,Y,Z,H as a type, Rz as one individual type
#  CNOT(control, target, Swap as a type
@everywhere function parse_rxry(n_qubits::Int,gate_name::String, para::Float64, idx::Int)
    chain(n_qubits,put(n_qubits, idx=>NoParams(gate_dict[gate_name](para))))
end

@everywhere function parse_rz(n_qubits::Int,gate_name::String, para::Float64, idx::Int)
    chain(n_qubits,put(n_qubits, idx=>gate_dict[gate_name](para)))
end

@everywhere function parse_single_qubit_gate(n_qubits::Int,gate_name::String, idx::Int)
    chain(n_qubits,put(n_qubits, idx=>gate_dict[gate_name]))
end

@everywhere function parse_two_qubit_gate(n_qubits::Int,gate_name::String, ctr::Int, target::Int)
    chain(n_qubits,control(ctr,target=>X))
end

@everywhere function get_energy(n_qubits::Int, n_electrons::Int, gate_list,params_list,para_key_list, hc)
    #println("Begin energy and gradient calculations")
    circuit, para_new_dt = gatelist_parse(n_qubits,n_electrons,gate_list,params_list,para_key_list)
    reg=zero_state(n_qubits) => circuit
    exp = expect(hc, reg) |> real
    exp
end

@everywhere function get_gradient(n_qubits::Int, n_electrons::Int, gate_list,params_list, para_key_list,hc)
    #println("Begin energy and gradient calculations")
    circuit, para_new_dt = gatelist_parse(n_qubits,n_electrons,gate_list,params_list,para_key_list)
    _, new_grad = expect'(hc, zero_state(n_qubits)=>circuit)
    # transform the corresponding gradients back to parameters.
    grad_compose = zeros(length(new_grad))
    grad_dt = Dict()
    for i=1:length(new_grad)
        grad_compose[i] = new_grad[i]
    end

    for i=1: length(grad_compose)
        para_dt = para_new_dt[i]
        para_dt_keys = keys(para_dt)
        for key in para_dt_keys
            if key in keys(grad_dt)
                grad_dt[key] +=  2*para_dt[key]*grad_compose[i]
            else
                grad_dt[key] =  2*para_dt[key]*grad_compose[i]
            end
        end
    end
    # get back to real parameters gradients
    #println("Begin here", grad_dt)
    grad = zeros(length(grad_dt))
    for i=1:length(grad_dt)
       grad[i]=grad_dt["p"*string(i-1)]
    end
    grad
end

@everywhere function get_energy_and_gradient(n_qubits::Int, n_electrons::Union{Int, Vector{Int}}, gate_list,params_list,para_key_list, hc)
    #println("Begin energy and gradient calculations")
#     println("para_key_list: ",para_key_list)
    circuit, para_new_dt = gatelist_parse(n_qubits,n_electrons,gate_list,params_list,para_key_list);
#     println("Number of parameters: ", nparameters(circuit))
    _, new_grad = expect'(hc, zero_state(n_qubits)=>circuit)
    # transform the corresponding gradients back to parameters.
    grad_compose = zeros(length(new_grad))
#     println("Length of grad_compose: ",length(grad_compose))
    grad_dt = Dict()
    for i=1:length(new_grad)
        grad_compose[i] = new_grad[i]
    end
    
    for i=1: length(grad_compose)
        para_dt = para_new_dt[i]
        para_dt_keys = keys(para_dt)
        for key in para_dt_keys
            if key in keys(grad_dt)
                grad_dt[key] +=  2*para_dt[key]*grad_compose[i]
            else
                grad_dt[key] =  2*para_dt[key]*grad_compose[i]
            end
        end
    end
    # get back to real parameters gradients
    #println("Begin here", grad_dt)
    
    # rewrite the following functions to enable one to one mapping, since in this particularly situation
    # the input parameters some part may lost , such as input 44 ->40, some part cancel outs
    grad = zeros(length(grad_dt))
    # for i=1:length(grad_dt)
    #    grad[i]=grad_dt["p"*string(i-1)]
    # end
#     println("grad_dt is: ",length(grad_dt))
    for i=1:length(grad_dt)
       grad[i]=grad_dt[para_key_list[i]]
    end
    #println("Finish updating gradients.", length(grad))
    reg=zero_state(n_qubits) |> circuit
    exp = expect(hc, reg) |> real
    exp, grad
end

@everywhere function simulate(params_list,para_key_list,n_qubits::Int, n_electrons::Union{Int,Vector{Int}}, gate_list, hc)
        energy, grad = get_energy_and_gradient(n_qubits, n_electrons, gate_list,params_list,para_key_list, hc)

end

## The following modules is to implement the functions.
# try to make this one parallel by first preprocessing the ranking parts.

function single_fermion_term_energy_cal(n_qubits, n_electrons, pauli_term, params, hamil)
    """calculate the optimal energy for this single pauli terms."""
    #pauli_ansatz =_transform2pauli([fermion_term])
    pauli_ansatz, _, key = pauli_term
    gate_list = op._pauli2circuit(pauli_ansatz)
    # key = fermion_term[1]
    para_key_list= [key]
    initia_para_list = params[key]
    res = minimize(simulate, initia_para_list, args=(para_key_list, n_qubits, n_electrons, gate_list, hamil), method="L-BFGS-B",jac=true)
    return key, res["fun"], res["x"]
end 


@everywhere function estimate_energy(n_qubits::Int, circuit, hamil)
    reg = zero_state(n_qubits) => circuit
    exp = expect(hamil, reg) |> real
    exp
end


@everywhere function convert2statevec(n_qubits::Int, circuit)
    reg = zero_state(n_qubits) |> circuit
end

@everywhere function estimate_energy_util(n_qubits::Int, reg, iele, jele, hamil)
#     reg = zero_state(n_qubits) => circuit
    exp = expect(hamil, reg) |> real
    (iele,jele), exp
end

@everywhere function estimate_energy_util(n_qubits::Int, reg, iele, jele, kele, lele, hamil)
    #key, hamil = single_key_hamil
#     reg = zero_state(n_qubits) => circuit
    exp = expect(hamil, reg) |> real
    #println("the exp is:", exp)
    (iele,jele,kele,lele), exp
end 

@everywhere function parallel_estimate_energy(para_list)
    res  = pmap((args)->estimate_energy_util(args...), para_list)
end


# For a single molecule
function Kp_UCCGSD_simulate_mol(mol, real_mol=false,kp=1)
    # mol = py"make_mol"(bond_len)
    terms, coefs = op.get_hamiltonian(mol)
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons
    
    # get the hamiltonian of this system
    hamil = get_fermion_hamiltonian(n_qubits,terms,coefs);    
    #This set up default by up and down then up and down pattern
    gate_list, params=py"construct_uccsd"(n_qubits,n_electrons,  -1)
#     gate_list, params = py"construct_kp_uccgsd"(n_qubits,n_electrons, -1,kp)
    #println("gate_list: ", gate_list)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    # perform the actual simulation
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons, gate_list, hamil), method="L-BFGS-B", jac=true)
    res["fun"] #, res["x"]
end

function restricted_UCCSD_simulate_mol(n_qubits,n_electrons_alpha_beta,hamil, real_mol=false)
    gate_list, params=op.construct_uccsd(n_qubits, n_electrons_alpha_beta,  -1)
    #gate_list, params = py"construct_kp_uccgsd"(n_qubits,n_electrons, -1,kp)
    #println("gate_list: ", gate_list)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons_alpha_beta, gate_list, hamil), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons_alpha_beta,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    # # res["fun"], res["x"]
    res["fun"], opt_circuit, res["x"], para_key_list
end


function restricted_UCCSD_simulate_mol_optimal(n_qubits,n_electrons_alpha_beta, hamil, opt_param, para_key_list)
    # just reuse the gate_list generation process
    gate_list, params_init=op.construct_uccsd(n_qubits, n_electrons_alpha_beta,  -1)
    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, opt_param, args=(para_key_list,n_qubits, n_electrons_alpha_beta, gate_list, hamil), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons_alpha_beta,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    # # res["fun"], res["x"]
    res["fun"], opt_circuit, res["x"], para_key_list
end



# For a single molecule
function Kp_UCCGSD_simulate_mol(n_qubits,n_electrons,hamil, real_mol=false,kp=1) 
    #This set up default by up and down then up and down pattern
    gate_list, params=op.construct_uccsd(n_qubits, n_electrons,  -1)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    res = op.wrap_minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons, gate_list, hamil),method="SLSQP", jac=true)
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list,res["x"],para_key_list);
    # println("Finish the overall optimization.")
    # # res["fun"], res["x"]
    res["fun"], opt_circuit,rex["x"],para_key_list
end


function unrestricted_UCCSD_simulate_mol_optimal(n_qubits,n_electrons_alpha_beta,hamil, opt_para, para_key_list)
    gate_list, params_init=op.construct_unrestricted_uccsd(n_qubits, n_electrons_alpha_beta,  -1)
    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons_alpha_beta, gate_list, hamil), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons_alpha_beta,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    res["fun"], opt_circuit, res["x"], para_key_list
end

function unrestricted_UCCSD_simulate_mol(n_qubits,n_electrons_alpha_beta,hamil, real_mol=false)
    gate_list, params=op.construct_unrestricted_uccsd(n_qubits, n_electrons_alpha_beta,  -1)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons_alpha_beta, gate_list, hamil), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons_alpha_beta,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    res["fun"], opt_circuit, res["x"], para_key_list
end

@everywhere function adapt_vqe_get_energy_and_gradient(qst, commutate_ls)
    op_nums = length(commutate_ls)
    new_grad = zeros(op_nums)
    for i = 1: op_nums #TODO this could be accelerate
        new_grad[i]=abs(real(expect(commutate_ls[i], qst))) # abosulte value of gradient direction.
    end
    return new_grad
end

@everywhere function single_adapt_vqe_get_energy_and_gradient(qst, commutate_item)
    grad = abs(real(expect(commutate_item, qst))) # abosulte value of gradient direction.
    return grad
end

@everywhere function parallel_adapt_vqe_get_energy_and_gradient(para_list)
    res  = pmap((args)->single_adapt_vqe_get_energy_and_gradient(args...), para_list)
end

@everywhere function adapt_vqe(n_qubits::Int, n_electrons::Union{Int, Vector{Int}}, max_iteration::Int, index_key_map, gate_list, circuit_params, para_key_list, hc, commutate_ls)
    #println("Begin energy and gradient calculations")
#     println("para_key_list: ",para_key_list)
    
    # Step 1: 
    qst=zero_state(n_qubits)
    circuitx = chain(n_qubits, repeat(X, [1:n_electrons]))#chain(n_qubits, put(1=>I2))
    qst = qst |> circuitx
    gate_list_new = []
    params_list_new = []
    para_key_new = []
    # enter into the while loop
    op_idx = []
    energy_ls = []
    opt_para = []
    index_key = []
    i = 0
    norm_value = 1
    time_ls = []
    while(i<max_iteration || norm_value < 1e-3 ) # some other criteria
        # Step2: Get gradient
        pre_energy = real(expect(hc,qst))
        push!(energy_ls, pre_energy)
#         new_grad = adapt_vqe_get_energy_and_gradient(qst, commutate_ls)
        qst_ls = repeat([qst], length(commutate_ls))
        start1 = Base.time_ns()
        new_grad = parallel_adapt_vqe_get_energy_and_gradient(tuple.(qst_ls, commutate_ls))
        end1 = Base.time_ns()
        push!(time_ls, end1-start1)
        norm_value = norm(new_grad)
        index = argmax(new_grad)# choose the largest gradient
        
        # each time add two
#         index_ls = sortperm(new_grad, rev=true)[1:2]; #choose the top 3
#         for index in index_ls
#             push!(op_idx, index) # save this for further use.
#         # Step 3: Add this one to the circuit
#             key = index_key_map[index]
#             if !(key in para_key_new)
#                 #else do not update the parameter list,since this share para
#                 params_list_new = vcat(params_list_new, circuit_params[index])
#                 para_key_new = vcat(para_key_new, index_key_map[index])
#             end
#             gate_list_new = vcat(gate_list_new, gate_list[index])
#         end
        
        push!(op_idx, index) # save this for further use.
        # Step 3: Add this one to the circuit
        key = index_key_map[index]
        if !(key in para_key_new) # this code means no same operator with different parametes.
            #else do not update the parameter list,since this share para
            params_list_new = vcat(params_list_new, circuit_params[index])
            para_key_new = vcat(para_key_new, index_key_map[index])
        else
            println("******Already enter here.******")
        end
        gate_list_new = vcat(gate_list_new, gate_list[index])
    
            
         # This will update any more
        # deal with degenerate parameters
        # Optimize the quantum circuit with those above circuit
        # Step4: With the gradient shared by the paramters
        # exp, grad = get_energy_and_gradient(n_qubits, n_electrons, gate_list_new, params_list_new,para_key_new, hc)
        println("params_list_new: ", params_list_new)
        println("param_key_new: ", para_key_new)
        println("Energy list: ", energy_ls)
        # Step5: Optimize with this grad, this step already including the previous step
        start2 = Base.time_ns()
        res= minimize(simulate, params_list_new, args=(para_key_new,n_qubits, n_electrons, gate_list_new, hc), method="L-BFGS-B", jac=true)
        end2 = Base.time_ns() # this part will dominant in the future.
        push!(time_ls, end2- start2)
        # Step 6 update the current qst with this optimized circuit
        opt_energy = res["fun"] #, 
        params_list_new = res["x"] # update para
        push!(opt_para, para_key_new)
        #TODO circuit_new
        # Be careful this gate_list_pare already includeing the hartree fock state
        opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list_new,params_list_new,para_key_new);
        qst = zero_state(n_qubits) |> opt_circuit # need to configure this circuit_new function 
        if abs(opt_energy-pre_energy)<=1e-6
            print("Enter into here.")
            print("opt_energy: ", opt_energy)
            print("pre_energy: ", pre_energy)
            break
        else
            i = i+1
        end
        println("Iteration: ", i)
    end
    
    return energy_ls, op_idx, time_ls
   
end


@everywhere function adapt_vqe_new(n_qubits::Int, n_electrons::Union{Int, Vector{Int}}, max_iteration::Int, index_key_map, gate_list, circuit_params, para_key_list, hc, commutate_ls)
    #println("Begin energy and gradient calculations")
#     println("para_key_list: ",para_key_list)
    
    # Step 1: 
    qst=zero_state(n_qubits)
    circuitx = chain(n_qubits, repeat(X, [1:n_electrons]))#chain(n_qubits, put(1=>I2))
    qst = qst |> circuitx
    gate_list_new = []
    params_list_new = []
    para_key_new = []
    # enter into the while loop
    op_idx = []
    energy_ls = []
    opt_para = []
    index_key = []
    i = 0
    norm_value = 1
    time_ls = []
    opt_circuit_ls=[]
    while(i<max_iteration || norm_value < 1e-3 ) # some other criteria
        # Step2: Get gradient
        pre_energy = real(expect(hc,qst))
        push!(energy_ls, pre_energy)
#         new_grad = adapt_vqe_get_energy_and_gradient(qst, commutate_ls)
        qst_ls = repeat([qst], length(commutate_ls))
        start1 = Base.time_ns()
        new_grad = parallel_adapt_vqe_get_energy_and_gradient(tuple.(qst_ls, commutate_ls))
        end1 = Base.time_ns()
        push!(time_ls, end1-start1)
        norm_value = norm(new_grad)
        index = argmax(new_grad)# choose the largest gradient
        
        # each time add two
#         index_ls = sortperm(new_grad, rev=true)[1:2]; #choose the top 3
#         for index in index_ls
#             push!(op_idx, index) # save this for further use.
#         # Step 3: Add this one to the circuit
#             key = index_key_map[index]
#             if !(key in para_key_new)
#                 #else do not update the parameter list,since this share para
#                 params_list_new = vcat(params_list_new, circuit_params[index])
#                 para_key_new = vcat(para_key_new, index_key_map[index])
#             end
#             gate_list_new = vcat(gate_list_new, gate_list[index])
#         end
        
        push!(op_idx, index) # save this for further use.
        # Step 3: Add this one to the circuit
        key = index_key_map[index]
        if !(key in para_key_new) # this code means no same operator with different parametes.
            #else do not update the parameter list,since this share para
            params_list_new = vcat(params_list_new, circuit_params[index])
            para_key_new = vcat(para_key_new, index_key_map[index])
        else
            println("******Already enter here.******")
        end
        gate_list_new = vcat(gate_list_new, gate_list[index])
    
            
         # This will update any more
        # deal with degenerate parameters
        # Optimize the quantum circuit with those above circuit
        # Step4: With the gradient shared by the paramters
        # exp, grad = get_energy_and_gradient(n_qubits, n_electrons, gate_list_new, params_list_new,para_key_new, hc)
#         println("params_list_new: ", params_list_new)
#         println("param_key_new: ", para_key_new)
#         println("Energy list: ", energy_ls)
        # Step5: Optimize with this grad, this step already including the previous step
        start2 = Base.time_ns()
        res= minimize(simulate, params_list_new, args=(para_key_new,n_qubits, n_electrons, gate_list_new, hc), method="L-BFGS-B", jac=true)
        end2 = Base.time_ns() # this part will dominant in the future.
        push!(time_ls, end2- start2)
        # Step 6 update the current qst with this optimized circuit
        opt_energy = res["fun"] #, 
        params_list_new = res["x"] # update para
        push!(opt_para, para_key_new)
        #TODO circuit_new
        # Be careful this gate_list_pare already includeing the hartree fock state
        opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list_new,params_list_new,para_key_new);
        push!(opt_circuit_ls, opt_circuit)
        qst = zero_state(n_qubits) |> opt_circuit # need to configure this circuit_new function 
        if abs(opt_energy-pre_energy)<=1e-6
            println("Enter into here.")
            println("opt_energy: ", opt_energy)
            println("pre_energy: ", pre_energy)
            break
        else
            i += 1
        end
        println("Iteration: ", i)
    end
    
    return energy_ls, last(opt_circuit_ls), op_idx, params_list_new, para_key_new
   
end

function restricted_ADAPT_VQE(n_qubits::Int, n_electrons::Union{Int, Vector}, hamil_julia, max_iteration::Int) # this from python input
    # step1: define from the artifical molecular for embedding sytem
    # and get the hamiltonian


    # step2: first generate the pool, params, and params map for further use 
    fermion_ansatz, params, params_pair_map = op._para_uccsd_generator_pool_new2(n_qubits, n_electrons, -1)

    num = convert(UInt32, length(fermion_ansatz)/2) # half is terms, half is the parameter label
    qubit_op = op.adapt_gradient_op_pool_list(fermion_ansatz[1:num]);
    qubit_op_julia_ls = []
    # Note the imaginary part can be deal preprocessing.
    for item in qubit_op
        qubit_julia = get_fermion_hamiltonian(n_qubits,item[1],item[2])
        push!(qubit_op_julia_ls, qubit_julia)
    end
    #step3: configure the circuit
    pauli_dt = op._transform2pauli_onebyone(fermion_ansatz);
    circuit_ls = op._pauli2circuit_onebyone(pauli_dt);

    index_key_map = Dict()
    circuit_params = []
    for i=1: num
        key = fermion_ansatz[i+num]
        index_key_map[i] = fermion_ansatz[i+num]
        push!(circuit_params, params[key])
    end

    # step4: construct the initial para and para_key_list
    npara = length(params)
    params_list = []
    para_key_list = []
    for i=1:npara
        key = 'p'*string(i-1) # python index start with 0, while julia start with 1
        push!(params_list, params[key])
        push!(para_key_list, key)
    end

    # step5: Preprocess the commutator part for the evaluation of the gradient
    commutate_ls = []
    for item in qubit_op_julia_ls
        push!(commutate_ls, hamil_julia*item-item*hamil_julia)
    end

    # Step 6: Construct the ADAPT-VQE ansatz in this case
#     start = Base.time_ns()
     energy_ls,opt_circuit, op_idx, params_list_new, para_key_new = adapt_vqe_new(n_qubits, n_electrons, max_iteration, index_key_map, circuit_ls, circuit_params, para_key_list, hamil_julia, commutate_ls)
#     finish = Base.time_ns()
#     print("Time cost: ", (finish-start)/1e9)
    energy_ls, opt_circuit, op_idx, params_list_new, para_key_new
   
end


function restricted_ADAPT_VQE_Optimal(n_qubits, n_electrons, op_idx, para_key_list, params_list,hamil_julia) 
    # step1: define from the artifical molecular for embedding sytem
    # and get the hamiltonian

#     hamil_julia = get_fermion_hamiltonian(n_qubits,hamil[1],hamil[2]);
    # step2: first generate the pool, params, and params map for further use 
    fermion_ansatz, params, params_pair_map = op._para_uccsd_generator_pool_new2(n_qubits, n_electrons, -1)

    #step3: configure the circuit
    pauli_dt = op._transform2pauli_onebyone(fermion_ansatz);
    circuit_ls = op._pauli2circuit_onebyone(pauli_dt);
    gate_list = []
    for item in op_idx
        gate_list = vcat(gate_list, circuit_ls[item])
    end
    
    # choose the selected list
    # with the corresponding index and value
    
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons, gate_list, hamil_julia), method="L-BFGS-B", jac=true)
    #Save the corresponding para and circuit.
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list,res["x"], para_key_list);
    println("Finish the overall optimization.")
    res["fun"], opt_circuit, op_idx, res["x"], para_key_list
end

function test_adapt_LiH()
    # step1:  take LiH as example
    bond_length = 1.0
    mol = op.make_mol(bond_length)

    # step2: obtain the Hamiltonian
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons

    hamil = op.get_hamiltonian(mol);# basically a tuple that contains the terms and coefficient
    hamil_julia = get_fermion_hamiltonian(n_qubits,hamil[1],hamil[2]);

    # step3: first generate the pool, params, and params map for further use 
    fermion_ansatz, params, params_pair_map = op._para_uccsd_generator_pool_new2(n_qubits, n_electrons, -1)

    num = convert(UInt32, length(fermion_ansatz)/2) # half is terms, half is the parameter label
    qubit_op = op.adapt_gradient_op_pool_list(fermion_ansatz[1:num]);
    qubit_op_julia_ls = []
    # Note the imaginary part can be deal preprocessing.
    for item in qubit_op
        qubit_julia = get_fermion_hamiltonian(n_qubits,item[1],item[2])
        push!(qubit_op_julia_ls, qubit_julia)
    end
    #step4: configure the circuit
    pauli_dt = op._transform2pauli_onebyone(fermion_ansatz);
    circuit_ls = op._pauli2circuit_onebyone(pauli_dt);

    index_key_map = Dict()
    circuit_params = []
    for i=1: num
        key = fermion_ansatz[i+num]
        index_key_map[i] = fermion_ansatz[i+num]
        push!(circuit_params, params[key])
    end

    # step5: construct the initial para and para_key_list
    npara = length(params)
    params_list = []
    para_key_list = []
    for i=1:npara
        key = 'p'*string(i-1) # python index start with 0, while julia start with 1
        push!(params_list, params[key])
        push!(para_key_list, key)
    end

    # step6: Preprocess the commutator part for the evaluation of the gradient
    commutate_ls = []
    for item in qubit_op_julia_ls
        push!(commutate_ls, hamil_julia*item-item*hamil_julia)
    end

    # Step 7: Construct the ADAPT-VQE ansatz in this case

    max_iteration = 50
    start = Base.time_ns()
    energy_ls, op_idx, time_ls=adapt_vqe(n_qubits, n_electrons, max_iteration, index_key_map, circuit_ls,  circuit_params, para_key_list, hamil_julia, commutate_ls)
    finish = Base.time_ns()
    print("Time cost: ", (finish-start)/1e9)
end

function test_uccsd_lih()
    begin1 = Base.time_ns()
    # step1:   take LiH as example, configure the gemotry.
    bond_length = 1.0
    mol = op.make_mol(bond_length)

    # step2: obtain the Hamiltonian
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons

    hamil = op.get_hamiltonian(mol);# basically a tuple that contains the terms and coefficient
    hamil_julia = get_fermion_hamiltonian(n_qubits,hamil[1],hamil[2]);

    # step3: construct the gate list and the parameter map in this case.
    gate_list, params=op.construct_uccsd(n_qubits, n_electrons,  -1)
    #gate_list, params = op.construct_kp_uccgsd(n_qubits,n_electrons, -1,kp)
    #println("gate_list: ", gate_list)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    # step4: perform the optimization in this step.
    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons, gate_list, hamil_julia), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    # step5: save the optimized the circuit for further use.
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    # # res["fun"], res["x"]
    # res["fun"], opt_circuit
    end1 = Base.time_ns()
    println("time cost is: ", (end1-begin1)/1e9)
end

function test_uccsd_lih2()
    begin1 = Base.time_ns()
    # step1:   take LiH as example, configure the gemotry.
    bond_length = 1.0
    mol = op.make_mol_NH3()

    # step2: obtain the Hamiltonian
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons

    hamil = op.get_hamiltonian(mol);# basically a tuple that contains the terms and coefficient
    hamil_julia = get_fermion_hamiltonian(n_qubits,hamil[1],hamil[2]);

    # step3: construct the gate list and the parameter map in this case.
    gate_list, params=op.construct_uccsd(n_qubits, n_electrons,  -1)
    #gate_list, params = op.construct_kp_uccgsd(n_qubits,n_electrons, -1,kp)
    #println("gate_list: ", gate_list)
    params_list = []
    para_key_list = []
    for (key, v) in params
        # println("key:", key)
        push!(para_key_list,key)
        push!(params_list, params[key])
    end

    # step4: perform the optimization in this step.
    # perform the actual simulation, change the n_electrons into n_electrons_alpha_beta
    res= minimize(simulate, params_list, args=(para_key_list,n_qubits, n_electrons, gate_list, hamil_julia), method="L-BFGS-B", jac=true)
    #res["fun"] #, res["x"]
    # To make sure whether we converge for this case.
    open("status.txt","a") do file
        println(file, res["nit"])
        println(file, res["message"])
        println(file, res["success"])
    end
    # step5: save the optimized the circuit for further use.
    opt_circuit, _ = gatelist_parse(n_qubits,n_electrons,gate_list,res["x"],para_key_list);
    println("Finish the overall optimization.")
    # # res["fun"], res["x"]
    # res["fun"], opt_circuit
    end1 = Base.time_ns()
    println("time cost is: ", (end1-begin1)/1e9)
end

# println("Begin NH3 this optimization.")
# test_uccsd_lih2()
# println("End NH3 this optimization.")
