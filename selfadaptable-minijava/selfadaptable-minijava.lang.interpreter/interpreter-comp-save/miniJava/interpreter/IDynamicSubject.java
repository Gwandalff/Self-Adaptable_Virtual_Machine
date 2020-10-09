package miniJava.interpreter;

import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;

/**
 * IDynamicSubject : interface for the AST node that will be targeted by dynamic modules
 * @author gjouneau
 *
 */
public interface IDynamicSubject {
	
	/**
	 * attach : link a dynamic module to a node
	 * @param dm : the dynamic module to link
	 */
	void attach(IDynamicModule dm);
	
	/**
	 * detach : link a dynamic module to a node
	 * @param dm : the dynamic module to remove
	 */
	void detach(IDynamicModule dm);
	
	/**
	 * notifyDynamicModulesBefore : call the {@link IDynamicModule#updateBefore(IDynamicSubject)} 
	 * of all the linked modules and collect the return values of the modules with AROUND 
	 * {@link IDynamicModule.Strategy}. If one or more of the AROUND module return false, then
	 * the semantic of the node is skipped.
	 * @return false if one or more of the AROUND modules return false, else true
	 */
	boolean notifyDynamicModulesBefore(Object[] args);
	
	/**
	 * notifyDynamicModulesAfter : call the {@link IDynamicModule#updateAfter(IDynamicSubject, Value)} 
	 * of all the linked modules and collect the return values of the modules. If a module returns true
	 * it call the {@link IDynamicModule#byPassResult()} method and save it as the new result.
	 * @param result : the actual result of the computation(can be null if no result or semantics skip)
	 * @return The last proposed result (from computation if the result wasn't override)
	 */
	Value notifyDynamicModulesAfter(Object[] args, Value result);
}
