package miniJava.interpreter;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

import miniJava.interpreter.miniJava.Value;

/**
 * DynamicAspect : Aspect that create the glue between the interpreter and the dynamic modules
 * @author gjouneau
 *
 */
@Aspect
public class DynamicAspect {

	/**
	 * aroundExpression : add calls to dynamic modules before and after the semantics execution.
	 * Moreover, allow to shortcut semantics from modules using the AROUND {@link IDynamicModule.Strategy}
	 * and override the result of the execution
	 * @param pjp : The joinpoint giving access to the context of the code injection(i.e. AST node executed)
	 * @return The value resulting of the computation or an override of this value
	 * @throws Throwable
	 
	@Around("execution(Value miniJava.interpreter.miniJava.impl.*.evaluateExpression(..))")
    public Value aroundExpression(ProceedingJoinPoint pjp) throws Throwable {
		IDynamicSubject node = (IDynamicSubject) pjp.getTarget();
		
		boolean doTheMethod = node.notifyDynamicModulesBefore(pjp.getArgs());
		
		Value out = null;
		if(doTheMethod){
			out = (Value) pjp.proceed();
		}
		
		out = node.notifyDynamicModulesAfter(pjp.getArgs(), ((Value) out));
		return out;
    }*/
	
	/**
	 * aroundStatement : add calls to dynamic modules before and after the semantics execution.
	 * Moreover, allow to shortcut semantics from modules using the AROUND {@link IDynamicModule.Strategy}
	 * @param pjp : The joinpoint giving access to the context of the code injection(i.e. AST node executed)
	 * @throws Throwable
	 */
	@Around("execution(void miniJava.interpreter.miniJava.impl.*.evaluateStatement(..))")
    public void aroundStatement(ProceedingJoinPoint pjp) throws Throwable {
		IDynamicSubject node = (IDynamicSubject) pjp.getTarget();
		
		boolean doTheMethod = node.notifyDynamicModulesBefore(pjp.getArgs());
		
		if(doTheMethod){
			pjp.proceed();
		}
		
		node.notifyDynamicModulesAfter(pjp.getArgs(), ((Value) null));
    }
	
	@Around("execution(void miniJava.interpreter.miniJava.impl.MethodCallImpl.call(..))")
    public void aroundCall(ProceedingJoinPoint pjp) throws Throwable {
		IDynamicSubject node = (IDynamicSubject) pjp.getTarget();
		
		boolean doTheMethod = node.notifyDynamicModulesBefore(pjp.getArgs());
		
		if(doTheMethod){
			pjp.proceed();
		}
		
		node.notifyDynamicModulesAfter(pjp.getArgs(), ((Value) null));
    }
}
