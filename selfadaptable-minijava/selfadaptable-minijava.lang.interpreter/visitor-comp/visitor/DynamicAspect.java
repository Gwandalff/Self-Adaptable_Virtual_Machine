package visitor;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

@Aspect
public class DynamicAspect {
 
	@Around("execution(* visitor.VisitorImplementation.*(..))")
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
		AcceptInterface node = (AcceptInterface) pjp.getArgs()[0];
		node.notifyDynamicModulesBefore();
		boolean doTheMethod = node.notifyDynamicModulesBefore();
		Object out = null;
		if(doTheMethod){
			out = pjp.proceed();
		}
		node.notifyDynamicModulesAfter();
		node.notifyDynamicModulesAfter();
		return out;
    }
}
