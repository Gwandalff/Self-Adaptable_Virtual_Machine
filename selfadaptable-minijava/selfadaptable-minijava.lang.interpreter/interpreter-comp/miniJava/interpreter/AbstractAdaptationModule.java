package miniJava.interpreter;

import java.util.List;

import miniJava.interpreter.decision.model.Goal;
import miniJava.interpreter.decision.model.Resource;
import miniJava.interpreter.decision.model.Softgoal;
import miniJava.interpreter.miniJava.Statement;
import miniJava.interpreter.miniJava.Value;

/**
 * IDynamicModule : Interface that define a dynamic module for the Dynamic Adaptive Pattern
 * @author gjouneau
 *
 */
public abstract class AbstractAdaptationModule {
	
	public static Goal moduleTradeOff = null;
	public static boolean active = true;
	
	public enum Strategy{
		BEFORE, AFTER, AROUND
	}
	/**
	 * updateBefore : function called before execution of the node semantics
	 * @param self : The AST node that will be executed (one of the {@link IDynamicModule#targetedNodes()})
	 * @return true if the node need to be evaluated false to shortcut it 
	 * (shortcut only if {@link IDynamicModule#callStrategy() is AROUND})
	 */
	public abstract boolean updateBefore(IAdaptationNode node, Object[] args);
	
	/**
	 * updateAfter : function called after execution of the node semantics
	 * @param self : The AST node that was executed (one of the {@link IDynamicModule#targetedNodes()})
	 * @param returned : The {@link Value} returned by the semantics
	 * @return true if you need to bypass the actual result 
	 * ({@link IDynamicModule#byPassResult()} will be called if true)
	 */
	public abstract boolean updateAfter(IAdaptationNode node, Object[] args, Value returned);
	
	/**
	 * byPassResult : function called when you bypass the result with 
	 * {@link IDynamicModule#updateAfter(IDynamicSubject, Value)}
	 * @return the Value you want to return (Warning : may be override by other modules)
	 */
	public abstract Value byPassResult();
	
	/**
	 * targetedNodes : return the array of nodes ID concerned by this module
	 * @return array of nodes ID (Access using Package.YOUR_NODE)
	 */
	public abstract boolean isTargetedNode(IAdaptationNode s);
	
	/**
	 * callStrategy : Choose the call strategy of the module. The strategy can be BEFORE or AFTER node execution,
	 * or AROUND(BEFORE & AFTER) allowing to choose if the semantics will be actually executed.
	 * @return The call strategy of the module
	 */
	public abstract Strategy callStrategy();
	
	public final void afterRegister() {
		AdaptationContext adaptationContext = FeedbackLoop.getAdaptationContext();
		
		init(adaptationContext);
		
		List<Resource> resources = adaptationContext.getResources();
		for (Resource resource : resources) {
			connectResource(resource);
		}
		
		String[] softgoalIDs = adaptationContext.softgoalIDs();
		for (int i = 0; i < softgoalIDs.length; i++) {
			Softgoal soft = new Softgoal(softgoalIDs[i]);
			moduleTradeOff.addContribution(soft, 1.0 / softgoalIDs.length);
			connectSoftGoal(soft);
		}
	}
	
	public final Goal getModuleTradeOff() {
		return moduleTradeOff;
	}
	
	public abstract void init(AdaptationContext adaptationContext);
	
	public abstract void connectSoftGoal(Softgoal softgoal);
	
	public abstract void connectResource(Resource resource);
	
	public final void setEnabled(boolean b) {
		active = b;
	}
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
