package miniJava.interpreter;

import miniJava.interpreter.miniJava.Value;

/**
 * IDynamicModule : Interface that define a dynamic module for the Dynamic Adaptive Pattern
 * @author gjouneau
 *
 */
public interface IDynamicModule {
	
	public enum Strategy{
		BEFORE, AFTER, AROUND
	}
	/**
	 * updateBefore : function called before execution of the node semantics
	 * @param self : The AST node that will be executed (one of the {@link IDynamicModule#targetedNodes()})
	 * @return true if the node need to be evaluated false to shortcut it 
	 * (shortcut only if {@link IDynamicModule#callStrategy() is AROUND})
	 */
	boolean updateBefore(IDynamicSubject self, Object[] args);
	
	/**
	 * updateAfter : function called after execution of the node semantics
	 * @param self : The AST node that was executed (one of the {@link IDynamicModule#targetedNodes()})
	 * @param returned : The {@link Value} returned by the semantics
	 * @return true if you need to bypass the actual result 
	 * ({@link IDynamicModule#byPassResult()} will be called if true)
	 */
	boolean updateAfter(IDynamicSubject self, Object[] args, Value returned);
	
	/**
	 * byPassResult : function called when you bypass the result with 
	 * {@link IDynamicModule#updateAfter(IDynamicSubject, Value)}
	 * @return the Value you want to return (Warning : may be override by other modules)
	 */
	Value byPassResult();
	
	/**
	 * targetedNodes : return the array of nodes ID concerned by this module
	 * @return array of nodes ID (Access using Package.YOUR_NODE)
	 */
	int[] targetedNodes();
	
	/**
	 * callStrategy : Choose the call strategy of the module. The strategy can be BEFORE or AFTER node execution,
	 * or AROUND(BEFORE & AFTER) allowing to choose if the semantics will be actually executed.
	 * @return The call strategy of the module
	 */
	Strategy callStrategy();
}



/*
 * @around(execution("VisitorInterface.*(..)"))
 * public void glue(ProceedingJoinPoint joinPoint){
 * 		EObject node = joinPoint.getASTNode();
 * 		node.notifyBefore();
 * 		boolean doTheMethod = node.notifyAroundStart();
 * 		if(doTheMethod){
 * 			joinPoint.proceed();
 * 		}
 * 		node.notifyAroundEnd();
 * 		node.notifyAfter();
 * }
 * 
 * */
